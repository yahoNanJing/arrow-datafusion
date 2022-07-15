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

use std::any::Any;
use std::borrow::Borrow;
use std::collections::hash_map::DefaultHasher;

use std::fmt::{Debug, Formatter};
use std::hash::BuildHasher;
use std::io::Cursor;
use std::sync::Arc;

use crate::{AggregateExpr, PhysicalExpr};
use arrow::array::{Array, ArrayRef, BinaryArray};
use arrow::datatypes::{DataType, Field};
use byteorder::{BigEndian, LittleEndian, ReadBytesExt};
use datafusion_common::{DataFusionError, Result, ScalarValue};
use datafusion_expr::Accumulator;
use hyperloglogplus::HyperLogLogPlus;

use crate::expressions::Literal;
use datafusion_common::DataFusionError::Internal;
use serde::{Deserialize, Serialize};

#[derive(Debug)]
pub struct KylinApproxDistinct {
    name: String,
    input_data_type: DataType,
    exprs: Vec<Arc<dyn PhysicalExpr>>,
    precision: u32,
}

impl KylinApproxDistinct {
    /// Create a new hllDistinct aggregate function.
    pub fn new(
        exprs: Vec<Arc<dyn PhysicalExpr>>,
        name: impl Into<String>,
        input_data_type: DataType,
    ) -> Result<Self> {
        if exprs.len() != 2 {
            return Err(DataFusionError::Plan(
                "KylinApproxDistinct require 2 args".to_string(),
            ));
        }
        // Extract the desired percentile literal
        let lit = exprs[1]
            .as_any()
            .downcast_ref::<Literal>()
            .ok_or_else(|| {
                DataFusionError::Internal(
                    "desired hll count argument must be float literal".to_string(),
                )
            })?
            .value();

        let precision = match lit {
            ScalarValue::UInt8(Some(q)) => *q as u32,
            ScalarValue::UInt16(Some(q)) => *q as u32,
            ScalarValue::UInt32(Some(q)) => *q as u32,
            got => return Err(DataFusionError::NotImplemented(format!(
                "Percentile value for 'KylinApproxDistinct' must be UInt8 UInt16 or UInt32 literal (got data type {})",
                got
            )))
        };

        Ok(Self {
            name: name.into(),
            input_data_type,
            exprs,
            precision,
        })
    }
}

impl AggregateExpr for KylinApproxDistinct {
    /// Return a reference to Any that can be used for downcasting
    fn as_any(&self) -> &dyn Any {
        self
    }

    /// the field of the final result of this aggregation.
    fn field(&self) -> Result<Field> {
        Ok(Field::new(&self.name, DataType::Int64, false))
    }

    fn create_accumulator(&self) -> Result<Box<dyn Accumulator>> {
        let accumulator: Box<dyn Accumulator> = match &self.input_data_type {
            DataType::Binary => {
                Box::new(KylinHyperloglogAccumulator::try_new(self.precision as u8))
            }
            other => {
                return Err(DataFusionError::NotImplemented(format!(
                    "Support for 'kylin_hll_distinct' for data type {} is not implemented",
                    other
                )));
            }
        };
        Ok(accumulator)
    }

    fn state_fields(&self) -> Result<Vec<Field>> {
        Ok(vec![Field::new(
            &format!("{} {}", self.name.as_str(), "kylin_hll_registers"),
            DataType::Binary,
            false,
        )])
    }

    fn expressions(&self) -> Vec<Arc<dyn PhysicalExpr>> {
        self.exprs.clone()
    }

    fn name(&self) -> &str {
        &self.name
    }
}

#[derive(Serialize, Deserialize)]
struct KylinHyperloglogAccumulator {
    hll: hyperloglogplus::HyperLogLogPlus<String, DefaultBuildHasher>,
    precision: u8,
}

impl KylinHyperloglogAccumulator {
    fn try_new(p: u8) -> Self {
        let mut hll = HyperLogLogPlus::new(p, DefaultBuildHasher {})
            .expect("Error create HyperLogLogPlus");
        hll.insert_direct_reg(0, 0);
        Self { hll, precision: p }
    }
}

impl Debug for KylinHyperloglogAccumulator {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        f.debug_struct("KylinHyperloglogAccumulator")
            .field("precision", &self.precision)
            .finish()
    }
}

impl Accumulator for KylinHyperloglogAccumulator {
    fn state(&self) -> Result<Vec<ScalarValue>> {
        let vec = serde_binary::to_vec(self.hll.borrow(), Default::default())
            .expect("fail to serialize in KylinHyperloglogAccumulator");
        Ok(vec![ScalarValue::Binary(Some(vec))])
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
                    let (mut keys, mut values) = try_deserialize_from_kylin(self.precision, bytes.expect("Impossibly got empty binary array from states in kylin_approx_distinct")).expect("fail to deserialize");
                    for _i in 0..keys.len() {
                        self.hll
                            .insert_direct_reg(keys.pop().unwrap(), values.pop().unwrap())
                    }
                }
            }
            e => {
                return Err(DataFusionError::Internal(format!(
                    "BITMAP_COUNT_DISTINCT is not expected to receive the type {:?}",
                    e
                )));
            }
        }
        Ok(())
    }

    fn merge_batch(&mut self, states: &[ArrayRef]) -> Result<()> {
        let binary_array = states[0].as_any().downcast_ref::<BinaryArray>().unwrap();
        for array in binary_array {
            let hll: hyperloglogplus::HyperLogLogPlus<String, DefaultBuildHasher> = serde_binary::from_vec(array.expect(
                "Impossibly got empty binary array from states in kylin_bitmap_distinct",
            ).to_owned(), Default::default())
                .expect("fail to deserialize HyperLogLogPlus");
            self.hll.merge(&hll).expect("fail to merge HyperLogLogPlus");
        }
        Ok(())
    }

    fn evaluate(&self) -> Result<ScalarValue> {
        Ok(ScalarValue::Int64(Some(self.hll.estimate_count() as i64)))
    }
}

#[derive(Serialize, Deserialize)]
struct DefaultBuildHasher;

impl BuildHasher for DefaultBuildHasher {
    type Hasher = DefaultHasher;

    fn build_hasher(&self) -> Self::Hasher {
        DefaultHasher::new()
    }
}

fn try_deserialize_from_kylin(p: u8, bytes: &[u8]) -> Result<(Vec<usize>, Vec<u32>)> {
    let mut cursor = Cursor::new(bytes);
    let schema = cursor.read_u8().unwrap();
    let index_len = kylin_get_register_index_size(p);
    let mut buckets = vec![];
    let mut hash_values = vec![];

    if schema == 0 {
        let size = kylin_read_vlong(&mut cursor).unwrap();
        for _i in 0..size {
            let key = kylin_read_unsigned(&mut cursor, index_len)?;
            let val = cursor.read_int::<BigEndian>(1)? as u32;
            //Some bucket will get -1 (not set)
            if key < (1 << p) {
                buckets.push(key as usize);
                hash_values.push(val);
            }
        }
    } else if schema == 1 {
        for i in 0..(1 << p) {
            let val = cursor.read_int::<BigEndian>(1)? as u32;
            buckets.push(i as usize);
            hash_values.push(val);
        }
    } else {
        return Err(Internal(format!("schema type {} not support", schema)));
    }
    Ok((buckets, hash_values))
}

fn kylin_read_unsigned(cursor: &mut Cursor<&[u8]>, index_len: u8) -> Result<u32> {
    Ok(cursor.read_uint::<LittleEndian>(index_len as usize)? as u32)
}

fn kylin_get_register_index_size(p: u8) -> u8 {
    (p - 1) / 8 + 1
}

fn kylin_read_vlong(cursor: &mut Cursor<&[u8]>) -> Result<i32> {
    let len = cursor.read_int::<BigEndian>(1)?;
    let len_after_decode = kylin_decode_vint_size(len as i8);
    if len_after_decode == 1 {
        return Ok(len as i32);
    }
    let mut res: i32 = 0;
    for _x in 0..len_after_decode - 1 {
        let b = cursor.read_int::<BigEndian>(1)? as i32;
        res <<= 8;
        res |= b & 0xFF;
    }
    if kylin_is_negative_vint(len as i8) {
        let mask: i32 = -1;
        Ok((res ^ mask) as i32)
    } else {
        Ok(res as i32)
    }
}

fn kylin_decode_vint_size(val: i8) -> i32 {
    if val >= -112 {
        return 1;
    } else if val < -120 {
        return (-119 - val) as i32;
    }
    (-111 - val) as i32
}

fn kylin_is_negative_vint(value: i8) -> bool {
    value < -120 || (value >= -112 && value < 0)
}
