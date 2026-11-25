//! Memory-mapped GGUF (v2/v3) reader: metadata, tensor index, zero-copy tensor bytes.

use memmap2::Mmap;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;

#[derive(Debug, Clone, PartialEq)]
pub enum Value {
    U64(u64),
    I64(i64),
    F64(f64),
    Bool(bool),
    Str(String),
    Arr(Vec<Value>),
}

#[allow(non_camel_case_types)]
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum GgmlType {
    F32,
    F16,
    Q8_0,
    Q4_K,
    Q5_K,
    Q6_K,
    BF16,
}

impl GgmlType {
    fn from_id(id: u32) -> Option<Self> {
        Some(match id {
            0 => Self::F32,
            1 => Self::F16,
            8 => Self::Q8_0,
            12 => Self::Q4_K,
            13 => Self::Q5_K,
            14 => Self::Q6_K,
            30 => Self::BF16,
            _ => return None,
        })
    }
    /// (elements per block, bytes per block)
    pub fn block(self) -> (u64, u64) {
        match self {
            Self::F32 => (1, 4),
            Self::F16 | Self::BF16 => (1, 2),
            Self::Q8_0 => (32, 34),
            Self::Q4_K => (256, 144),
            Self::Q5_K => (256, 176),
            Self::Q6_K => (256, 210),
        }
    }
}

#[derive(Debug, Clone)]
pub struct TensorInfo {
    pub name: String,
    /// ggml order: dims[0] is the contiguous (row) dimension.
    pub dims: Vec<u64>,
    pub ty: GgmlType,
    /// Absolute byte offset into the file.
    pub offset: usize,
    pub size: usize,
}

pub struct Gguf {
    mmap: Mmap,
    pub version: u32,
    pub meta: HashMap<String, Value>,
    pub tensors: Vec<TensorInfo>,
    index: HashMap<String, usize>,
}

struct Reader<'a> {
    b: &'a [u8],
    pos: usize,
}

type Res<T> = Result<T, String>;

impl<'a> Reader<'a> {
    fn take(&mut self, n: usize) -> Res<&'a [u8]> {
        let end = self.pos.checked_add(n).filter(|&e| e <= self.b.len()).ok_or("GGUF truncated")?;
        let s = &self.b[self.pos..end];
        self.pos = end;
        Ok(s)
    }
    fn u32(&mut self) -> Res<u32> {
        Ok(u32::from_le_bytes(self.take(4)?.try_into().unwrap()))
    }
    fn u64(&mut self) -> Res<u64> {
        Ok(u64::from_le_bytes(self.take(8)?.try_into().unwrap()))
    }
    fn str(&mut self) -> Res<String> {
        let n = self.u64()? as usize;
        String::from_utf8(self.take(n)?.to_vec()).map_err(|e| format!("GGUF string not UTF-8: {e}"))
    }
    fn value(&mut self, ty: u32) -> Res<Value> {
        let le = |s: &[u8]| {
            let mut a = [0u8; 8];
            a[..s.len()].copy_from_slice(s);
            a
        };
        Ok(match ty {
            0 => Value::U64(self.take(1)?[0] as u64),
            1 => Value::I64(self.take(1)?[0] as i8 as i64),
            2 => Value::U64(u16::from_le_bytes(self.take(2)?.try_into().unwrap()) as u64),
            3 => Value::I64(i16::from_le_bytes(self.take(2)?.try_into().unwrap()) as i64),
            4 => Value::U64(self.u32()? as u64),
            5 => Value::I64(self.u32()? as i32 as i64),
            6 => Value::F64(f32::from_le_bytes(self.take(4)?.try_into().unwrap()) as f64),
            7 => Value::Bool(self.take(1)?[0] != 0),
            8 => Value::Str(self.str()?),
            9 => {
                let et = self.u32()?;
                let n = self.u64()? as usize;
                let mut v = Vec::with_capacity(n.min(1 << 20));
                for _ in 0..n {
                    v.push(self.value(et)?);
                }
                Value::Arr(v)
            }
            10 => Value::U64(self.u64()?),
            11 => Value::I64(i64::from_le_bytes(le(self.take(8)?))),
            12 => Value::F64(f64::from_le_bytes(self.take(8)?.try_into().unwrap())),
            t => return Err(format!("unknown GGUF value type {t}")),
        })
    }
}

impl Gguf {
    pub fn open(path: impl AsRef<Path>) -> Res<Self> {
        let path = path.as_ref();
        let file = File::open(path).map_err(|e| format!("{}: {e}", path.display()))?;
        // SAFETY: the model file is treated as read-only for the life of the process.
        let mmap = unsafe { Mmap::map(&file) }.map_err(|e| format!("mmap {}: {e}", path.display()))?;

        let mut r = Reader { b: &mmap, pos: 0 };
        if r.take(4)? != b"GGUF" {
            return Err(format!("{}: not a GGUF file", path.display()));
        }
        let version = r.u32()?;
        if !(2..=3).contains(&version) {
            return Err(format!("unsupported GGUF version {version}"));
        }
        let n_tensors = r.u64()? as usize;
        let n_kv = r.u64()? as usize;

        let mut meta = HashMap::new();
        for _ in 0..n_kv {
            let k = r.str()?;
            let ty = r.u32()?;
            meta.insert(k, r.value(ty)?);
        }

        let mut raw = Vec::with_capacity(n_tensors);
        for _ in 0..n_tensors {
            let name = r.str()?;
            let nd = r.u32()? as usize;
            let dims = (0..nd).map(|_| r.u64()).collect::<Res<Vec<_>>>()?;
            let tid = r.u32()?;
            let ty = GgmlType::from_id(tid).ok_or_else(|| format!("{name}: unsupported ggml type {tid}"))?;
            raw.push((name, dims, ty, r.u64()?));
        }

        let align = match meta.get("general.alignment") {
            Some(Value::U64(a)) => *a as usize,
            _ => 32,
        };
        let base = r.pos.div_ceil(align) * align;

        let mut tensors = Vec::with_capacity(n_tensors);
        let mut index = HashMap::with_capacity(n_tensors);
        for (name, dims, ty, off) in raw {
            let (be, bb) = ty.block();
            let n: u64 = dims.iter().product();
            if dims[0] % be != 0 {
                return Err(format!("{name}: row {} not a multiple of block {be}", dims[0]));
            }
            let size = (n / be * bb) as usize;
            let offset = base + off as usize;
            if offset + size > mmap.len() {
                return Err(format!("{name}: data past end of file"));
            }
            index.insert(name.clone(), tensors.len());
            tensors.push(TensorInfo { name, dims, ty, offset, size });
        }

        Ok(Self { mmap, version, meta, tensors, index })
    }

    pub fn tensor(&self, name: &str) -> Option<&TensorInfo> {
        self.index.get(name).map(|&i| &self.tensors[i])
    }

    pub fn data(&self, t: &TensorInfo) -> &[u8] {
        &self.mmap[t.offset..t.offset + t.size]
    }

    /// Zero-copy view of an F32 tensor.
    pub fn f32s(&self, t: &TensorInfo) -> &[f32] {
        assert_eq!(t.ty, GgmlType::F32, "{} is not F32", t.name);
        let b = self.data(t);
        // SAFETY: mmap is page-aligned and offsets are 32-byte aligned; f32 has no invalid bit patterns.
        let (pre, f, _) = unsafe { b.align_to::<f32>() };
        assert!(pre.is_empty());
        f
    }

    pub fn str(&self, k: &str) -> Option<&str> {
        match self.meta.get(k)? {
            Value::Str(s) => Some(s),
            _ => None,
        }
    }

    pub fn u64(&self, k: &str) -> Option<u64> {
        match self.meta.get(k)? {
            Value::U64(v) => Some(*v),
            Value::I64(v) => u64::try_from(*v).ok(),
            _ => None,
        }
    }

    pub fn f64(&self, k: &str) -> Option<f64> {
        match self.meta.get(k)? {
            Value::F64(v) => Some(*v),
            _ => None,
        }
    }

    pub fn arr(&self, k: &str) -> Option<&[Value]> {
        match self.meta.get(k)? {
            Value::Arr(v) => Some(v),
            _ => None,
        }
    }
}
