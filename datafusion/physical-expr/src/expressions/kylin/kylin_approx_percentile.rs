// Licensed to the Apache Software Foundation (ASF) under one
// or more contributor license agreements.  See the NOTICE file
// distributed with this work for additional information
// regarding copyright ownership.  The ASF licenses this file
// to you under the Apache License, Version 2.0 (the
// "License"); you may not use this file except in compliance
// with the License.  You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing,
// software distributed under the License is distributed on an
// "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
// KIND, either express or implied.  See the License for the
// specific language governing permissions and limitations
// under the License.

use crate::aggregate::approx_percentile_cont::ApproxPercentileAccumulator;
use crate::expressions::ApproxPercentileCont;
use crate::{
    aggregate::tdigest::{Centroid, TDigest, DEFAULT_MAX_SIZE},
    AggregateExpr, PhysicalExpr,
};
use arrow::{
    array::ArrayRef,
    datatypes::{DataType, Field},
};

use datafusion_common::ScalarValue;
use datafusion_common::{DataFusionError, Result};
use datafusion_expr::Accumulator;

use arrow::array::{Array, BinaryArray};
use byteorder::{BigEndian, ReadBytesExt};
use std::io::Cursor;
use std::{any::Any, sync::Arc};

static VERBOSE_ENCODING: u32 = 1;
static SMALL_ENCODING: u32 = 2;

/// KYLIN_APPROX_PERCENTILE_CONT aggregate expression
#[derive(Debug)]
pub struct KylinApproxPercentile {
    approx_percentile_cont: ApproxPercentileCont,
    exprs: Vec<Arc<dyn PhysicalExpr>>,
}

impl KylinApproxPercentile {
    /// Create a new [`KylinApproxPercentileCont`] aggregate function.
    pub fn new(
        expr: Vec<Arc<dyn PhysicalExpr>>,
        name: impl Into<String>,
        return_type: DataType,
    ) -> Result<Self> {
        // Arguments should be [ColumnExpr, DesiredPercentileLiteral]
        if expr.len() != 2 {
            return Err(DataFusionError::Plan(
                "KylinApproxPercentile require 2 args".to_string(),
            ));
        }
        let sub_expr = vec![expr[0].clone(), expr[1].clone()];
        let approx_percentile_cont =
            ApproxPercentileCont::new(sub_expr, name, return_type)?;

        Ok(Self {
            approx_percentile_cont,
            exprs: expr.clone(),
        })
    }
}

impl AggregateExpr for KylinApproxPercentile {
    fn as_any(&self) -> &dyn Any {
        self
    }

    fn field(&self) -> Result<Field> {
        self.approx_percentile_cont.field()
    }

    #[allow(rustdoc::private_intra_doc_links)]
    /// See [`TDigest::to_scalar_state()`] for a description of the serialised
    /// state.
    fn state_fields(&self) -> Result<Vec<Field>> {
        self.approx_percentile_cont.state_fields()
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.exprs.clone()
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        let approx_percentile_cont_accumulator = ApproxPercentileAccumulator::new(
            self.approx_percentile_cont.get_percentile(),
            DataType::Float64,
        );
        let accumulator =
            KylinApproxPercentileAccumulator::new(approx_percentile_cont_accumulator);
        Ok(Box::new(accumulator))
    }

    fn name(&self) -> &str {
        self.approx_percentile_cont.name()
    }
}

#[derive(Debug)]
pub struct KylinApproxPercentileAccumulator {
    approx_percentile_cont_accumulator: ApproxPercentileAccumulator,
}

impl KylinApproxPercentileAccumulator {
    pub fn new(approx_percentile_cont_accumulator: ApproxPercentileAccumulator) -> Self {
        Self {
            approx_percentile_cont_accumulator,
        }
    }
}

impl Accumulator for KylinApproxPercentileAccumulator {
    fn state(&self) -> Result<Vec<ScalarValue>> {
        self.approx_percentile_cont_accumulator.state()
    }

    fn update_batch(&mut self, values: &[ArrayRef]) -> Result<()> {
        let value = &values[0];
        if value.is_empty() {
            return Ok(());
        }
        match value.data_type() {
            DataType::Binary => {
                let array = value.as_any().downcast_ref::<BinaryArray>().unwrap();
                for bytes in array {
                    let (means, weights) = try_deserialize_from_kylin(bytes.expect("Impossibly got empty binary array from states in kylin_bitmap_distinct")).expect("fail to deserialize");
                    debug_assert_eq!(
                        means.len(),
                        weights.len(),
                        "invalid number of values in means and weights"
                    );

                    let mut digests: Vec<TDigest> = vec![];
                    for (mean, weight) in means.iter().zip(weights.iter()) {
                        digests.push(TDigest::new_with_centroid(
                            DEFAULT_MAX_SIZE,
                            Centroid::new(*mean, *weight),
                        ))
                    }
                    self.approx_percentile_cont_accumulator
                        .merge_digests(&digests);
                }
                Ok(())
            }
            e => {
                return Err(DataFusionError::Internal(format!(
                    " KylinApproxPercentile is not expected to receive the type {:?}",
                    e
                )));
            }
        }
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        self.approx_percentile_cont_accumulator.evaluate()
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        self.approx_percentile_cont_accumulator
            .merge_batch(states)?;

        Ok(())
    }
}

fn try_deserialize_from_kylin(bytes: &[u8]) -> Result<(Vec<f64>, Vec<f64>)> {
    let mut cursor = Cursor::new(bytes);
    let i = cursor.read_int::<BigEndian>(4)? as u32;
    if i == VERBOSE_ENCODING {
        let _compression = cursor.read_f64::<BigEndian>()? as f64;
        let size = cursor.read_int::<BigEndian>(4)? as usize;
        let mut means: Vec<f64> = Vec::with_capacity(size);
        let mut weight: Vec<f64> = Vec::with_capacity(size);

        for _i in 0..size {
            means.push(cursor.read_f64::<BigEndian>()?);
        }
        for _i in 0..size {
            weight.push(cursor.read_u32::<BigEndian>()? as f64);
        }
        Ok((means, weight))
    } else if i == SMALL_ENCODING {
        let _compression = cursor.read_f64::<BigEndian>()? as f64;
        let size = cursor.read_int::<BigEndian>(4)? as usize;
        let mut means: Vec<f64> = Vec::with_capacity(size);
        let mut weight: Vec<f64> = Vec::with_capacity(size);

        let mut x: f64 = 0.0;
        for _i in 0..size {
            let delta = cursor.read_f32::<BigEndian>()?;
            x += delta as f64;
            means.push(x);
        }
        for _i in 0..size {
            weight.push(var_byte_decode(&mut cursor).unwrap() as f64);
        }
        Ok((means, weight))
    } else {
        Err(DataFusionError::NotImplemented(
            "Not implement encoding in kylin_percentile".to_string(),
        ))
    }
}

pub(crate) fn var_byte_decode(cursor: &mut Cursor<&[u8]>) -> Result<u32> {
    let mut v = cursor.read_u8().unwrap() as u32;
    let mut z = 0x7f & v as u32;
    let mut shift = 7;
    while (v & 0x80) != 0 {
        if shift > 28 {
            return Err(DataFusionError::Internal(
                "Fail in var_byte_decode".to_string(),
            ));
        }
        v = cursor.read_u8().unwrap() as u32;
        z += (v & 0x7f) << shift;
        shift += 7;
    }
    Ok(z)
}
