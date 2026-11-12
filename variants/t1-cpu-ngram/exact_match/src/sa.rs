//! EXACT_BACKEND=sa (cargo feature `sa`): the exact order-2..8 counts of track_1_short/exact_counts.py from a suffix
//! array instead of the GPU scan (infini-gram style, Liu et al. 2024).
//!
//! One SuffixCounts holds one corpus share (the concatenation of its shard files, as ExactCounts reads them) reversed,
//! and the suffix array of that reversed text (libsais16; 32-bit entries below 2^31 tokens, else 40-bit packed ones).
//! A gram's corpus count is the size of the SA range whose suffixes start with the reversed gram, so the grams ending
//! at a query position, read backwards from it, narrow one range step by step: the context chain (tok[i], tok[i-1],
//! ...) gives N for context lengths 2..8, the target chain (y, tok[i], tok[i-1], ...) gives C for lengths 3..9.
//!
//! The counts are those of ExactCounts._count, bit for bit (up to its 64-bit hash collisions), including where that
//! code returns 0 although the gram occurs: a context's count is read from the query table, which holds only the grams
//! some query inserted (each query's grams that end with its target). The trainer's queries are contiguous, so query
//! i - 1 inserts query i's contexts; where that does not hold, the inserted grams are checked exactly (`fix_uninserted`).
use numpy::{PyReadonlyArray1, PyReadwriteArray2};
use pyo3::{exceptions::PyValueError, prelude::*};
use rayon::prelude::*;
use std::collections::HashSet;
use std::io::Read;

pub const MAXO: usize = 8; // exact_counts.MAX_ORDER
pub const COLS: usize = 2 * MAXO - 2; // ExactCounts.out's columns: N for lengths 2..8, then C for context lengths 2..8
const SYMS: usize = 1 << 16;

/// The suffix array: 32-bit entries, or 40 bits (5 little-endian bytes) per entry from 2^31 tokens on.
pub enum Sa {
    I32(Vec<i32>),
    P40(Vec<u8>),
}

impl Sa {
    #[inline(always)]
    fn get(&self, k: usize) -> usize {
        match self {
            Sa::I32(v) => v[k] as usize,
            Sa::P40(b) => {
                let o = 5 * k;
                let mut w = [0u8; 8];
                w[..5].copy_from_slice(&b[o..o + 5]);
                u64::from_le_bytes(w) as usize
            }
        }
    }
    fn bytes(&self) -> usize {
        match self {
            Sa::I32(v) => 4 * v.len(),
            Sa::P40(b) => b.len(),
        }
    }
}

pub struct Core {
    pub text: Vec<u16>, // the share, reversed
    pub sa: Sa,
    pub bucket: Vec<u64>, // bucket[c]..bucket[c + 1]: the SA range of the suffixes starting with c
}

/// libsais16 over `text` into a fresh SA (threads > 1 only with the `sa-openmp` feature; else one thread).
fn suffix_sort(text: &[u16], threads: usize, force_p40: bool) -> Result<Sa, String> {
    let n = text.len();
    #[cfg(not(feature = "sa-openmp"))]
    let _ = threads;
    if n < (1usize << 31) && !force_p40 {
        let mut sa = vec![0i32; n];
        #[cfg(feature = "sa-openmp")]
        let rc = unsafe {
            libsais_sys::libsais16::libsais16_omp(text.as_ptr(), sa.as_mut_ptr(), n as i32, 0, std::ptr::null_mut(), threads as i32)
        };
        #[cfg(not(feature = "sa-openmp"))]
        let rc = unsafe { libsais_sys::libsais16::libsais16(text.as_ptr(), sa.as_mut_ptr(), n as i32, 0, std::ptr::null_mut()) };
        if rc != 0 {
            return Err(format!("libsais16 failed ({rc}) on {n} tokens"));
        }
        return Ok(Sa::I32(sa));
    }
    if n >= (1usize << 40) {
        return Err(format!("{n} tokens: over the 40-bit suffix array's 2^40"));
    }
    let mut sa = vec![0i64; n];
    #[cfg(feature = "sa-openmp")]
    let rc = unsafe {
        libsais_sys::libsais16x64::libsais16x64_omp(text.as_ptr(), sa.as_mut_ptr(), n as i64, 0, std::ptr::null_mut(), threads as i32)
    };
    #[cfg(not(feature = "sa-openmp"))]
    let rc = unsafe { libsais_sys::libsais16x64::libsais16x64(text.as_ptr(), sa.as_mut_ptr(), n as i64, 0, std::ptr::null_mut()) };
    if rc != 0 {
        return Err(format!("libsais16x64 failed ({rc}) on {n} tokens"));
    }
    let mut packed = vec![0u8; 5 * n]; // peak 2 + 8 + 5 bytes per token while packing
    packed.par_chunks_mut(5 << 16).zip(sa.par_chunks(1 << 16)).for_each(|(dst, src)| {
        for (d, &s) in dst.chunks_exact_mut(5).zip(src) {
            d.copy_from_slice(&(s as u64).to_le_bytes()[..5]);
        }
    });
    Ok(Sa::P40(packed))
}

impl Core {
    /// The index of `text` (the share in reading order): reversed in place, then suffix sorted.
    pub fn build(mut text: Vec<u16>, threads: usize, force_p40: bool) -> Result<Self, String> {
        text.reverse();
        let mut bucket = vec![0u64; SYMS + 1];
        for &t in &text {
            bucket[t as usize + 1] += 1;
        }
        for c in 0..SYMS {
            bucket[c + 1] += bucket[c];
        }
        let sa = suffix_sort(&text, threads, force_p40)?;
        Ok(Self { text, sa, bucket })
    }

    /// The symbol at depth `d` of the k-th suffix, -1 past the text's end (a shorter suffix sorts first).
    #[inline(always)]
    fn sym(&self, k: usize, d: usize) -> i32 {
        let p = self.sa.get(k) + d;
        if p < self.text.len() { self.text[p] as i32 } else { -1 }
    }

    /// Within [lo, hi) (suffixes sharing their first d symbols), the range whose symbol at depth d is c.
    #[inline]
    fn narrow(&self, lo: usize, hi: usize, d: usize, c: i32) -> (usize, usize) {
        let (mut a, mut b) = (lo, hi);
        while a < b {
            let m = a + (b - a) / 2;
            if self.sym(m, d) < c { a = m + 1 } else { b = m }
        }
        let first = a;
        b = hi;
        while a < b {
            let m = a + (b - a) / 2;
            if self.sym(m, d) <= c { a = m + 1 } else { b = m }
        }
        (first, a)
    }

    #[inline]
    fn first(&self, c: i32) -> (usize, usize) {
        (self.bucket[c as usize] as usize, self.bucket[c as usize + 1] as usize)
    }

    /// The occurrences of `gram` (reading order) in the share.
    pub fn count(&self, gram: &[i32]) -> u64 {
        let Some((&last, rest)) = gram.split_last() else { return self.text.len() as u64 };
        let (mut lo, mut hi) = self.first(last);
        for (d, &c) in rest.iter().rev().enumerate() {
            if lo == hi {
                break;
            }
            (lo, hi) = self.narrow(lo, hi, d + 1, c);
        }
        (hi - lo) as u64
    }

    /// One query's row (zeros past its segment start), as _find_kernel writes it before the inserted-gram rule.
    #[inline]
    fn row(&self, tok: &[i32], tgt: &[i32], start: &[i64], i: usize, row: &mut [i32]) {
        row.fill(0);
        let s = start[i];
        // the context chain: tok[i], tok[i - 1], ...; N of length m at column m - 2
        let (mut lo, mut hi) = self.first(tok[i]);
        for m in 2..=MAXO {
            let j = i as i64 - (m as i64 - 1);
            if j < s || lo == hi {
                break;
            }
            (lo, hi) = self.narrow(lo, hi, m - 1, tok[j as usize]);
            row[m - 2] = (hi - lo) as u32 as i32;
        }
        // the target chain: y, tok[i], tok[i - 1], ...; C of context length m at column MAXO + m - 3
        let (mut lo, mut hi) = self.first(tgt[i]);
        (lo, hi) = self.narrow(lo, hi, 1, tok[i]);
        for m in 2..=MAXO {
            let j = i as i64 - (m as i64 - 1);
            if j < s || lo == hi {
                break;
            }
            (lo, hi) = self.narrow(lo, hi, m, tok[j as usize]);
            row[MAXO + m - 3] = (hi - lo) as u32 as i32;
        }
    }

    /// The counts of every query into out [n, COLS] on the current rayon pool.
    pub fn counts(&self, tok: &[i32], tgt: &[i32], start: &[i64], out: &mut [i32]) -> Result<(), String> {
        let n = tok.len();
        if tgt.len() != n || start.len() != n || out.len() != n * COLS {
            return Err(format!("lengths: tok {n}, tgt {}, start {}, out {} (want {})", tgt.len(), start.len(), out.len(), n * COLS));
        }
        if let Some(i) = (0..n).find(|&i| !(0..SYMS as i32).contains(&tok[i]) || !(0..SYMS as i32).contains(&tgt[i]) || start[i] > i as i64) {
            return Err(format!("query {i}: token {} target {} start {} (tokens 0..65535, start <= position)", tok[i], tgt[i], start[i]));
        }
        out.par_chunks_mut(COLS * 1024).enumerate().for_each(|(b, rows)| {
            for (r, row) in rows.chunks_exact_mut(COLS).enumerate() {
                self.row(tok, tgt, start, b * 1024 + r, row);
            }
        });
        fix_uninserted(tok, tgt, start, out);
        Ok(())
    }

    /// The longest suffix of each query's context (ending at tok[i], inside its segment, at most max_len tokens) that
    /// occurs in the share: its length and count (count: the share's length at length 0). The infini-gram backoff.
    pub fn longest(&self, tok: &[i32], start: &[i64], max_len: usize, len: &mut [i32], cnt: &mut [i64]) {
        len.par_iter_mut().zip(cnt.par_iter_mut()).enumerate().for_each(|(i, (l, c))| {
            let (mut lo, mut hi) = (0usize, self.text.len());
            let mut m = 0usize;
            while m < max_len {
                let j = i as i64 - m as i64;
                if j < start[i] {
                    break;
                }
                let (a, b) = if m == 0 { self.first(tok[j as usize]) } else { self.narrow(lo, hi, m, tok[j as usize]) };
                if a == b {
                    break;
                }
                (lo, hi, m) = (a, b, m + 1);
            }
            (*l, *c) = (m as i32, (hi - lo) as i64);
        });
    }

    pub fn memory(&self) -> usize {
        2 * self.text.len() + self.sa.bytes() + 8 * self.bucket.len()
    }
}

/// Key of a gram of at most 8 tokens: (length, the tokens' 16 bits each).
fn key(gram: &[i32]) -> (u8, u128) {
    (gram.len() as u8, gram.iter().fold(0u128, |k, &t| k << 16 | t as u16 as u128))
}

/// _find_kernel reads a context's count from the query table, so a context no query inserted counts 0. Query j
/// inserts tok[j - m + 1 ..= j] + tgt[j] for m = 1..8 inside its segment; query i - 1 inserts query i's m-token context
/// whenever tgt[i - 1] == tok[i] and the context starts inside query i - 1's segment. The other nonzero contexts are
/// looked for among every query's inserted grams, and zeroed where none matches.
pub fn fix_uninserted(tok: &[i32], tgt: &[i32], start: &[i64], out: &mut [i32]) {
    let n = tok.len();
    let covered = |i: usize, m: usize| i >= 1 && tgt[i - 1] == tok[i] && i as i64 - (m as i64 - 1) >= start[i - 1];
    let counts: &[i32] = out;
    let todo: Vec<(usize, usize)> = (0..n)
        .into_par_iter()
        .flat_map_iter(|i| (2..=MAXO).filter(move |&m| counts[i * COLS + m - 2] != 0 && !covered(i, m)).map(move |m| (i, m)))
        .collect();
    if todo.is_empty() {
        return;
    }
    let want: HashSet<(u8, u128)> = todo.iter().map(|&(i, m)| key(&tok[i + 1 - m..=i])).collect();
    let found: HashSet<(u8, u128)> = (0..n)
        .into_par_iter()
        .fold(HashSet::new, |mut acc, j| {
            let mut g = [0i32; MAXO];
            for m in 1..MAXO {
                // query j's (m + 1)-token gram; contexts are at most MAXO tokens
                if (j as i64) - (m as i64 - 1) < start[j] {
                    break;
                }
                g[..m].copy_from_slice(&tok[j + 1 - m..=j]);
                g[m] = tgt[j];
                let k = key(&g[..=m]);
                if want.contains(&k) {
                    acc.insert(k);
                }
            }
            acc
        })
        .reduce(HashSet::new, |mut a, b| {
            a.extend(b);
            a
        });
    for (i, m) in todo {
        if !found.contains(&key(&tok[i + 1 - m..=i])) {
            out[i * COLS + m - 2] = 0;
        }
    }
}

/// The share as ExactCounts reads it: each file's tokens after its 1024-byte header, concatenated in order.
fn read_share(files: &[String]) -> PyResult<Vec<u16>> {
    let mut sizes = Vec::with_capacity(files.len());
    for f in files {
        sizes.push(crate::open_shard(f)?.1);
    }
    let mut text = vec![0u16; sizes.iter().sum()];
    let mut off = 0;
    for (f, &n) in files.iter().zip(&sizes) {
        let (mut file, _) = crate::open_shard(f)?;
        let dst = unsafe { std::slice::from_raw_parts_mut(text[off..off + n].as_mut_ptr() as *mut u8, 2 * n) };
        file.read_exact(dst)?;
        off += n;
    }
    if cfg!(target_endian = "big") {
        text.iter_mut().for_each(|t| *t = u16::from_le(*t));
    }
    Ok(text)
}

/// EXACT_BACKEND=sa: one corpus share's suffix array and its count queries.
#[pyclass(module = "exact_match")]
pub struct SuffixCounts {
    core: Core,
    pool: rayon::ThreadPool,
}

#[pymethods]
impl SuffixCounts {
    /// From shard files (concatenated in the given order, headers skipped), or from `tokens` (a uint16 array) when
    /// `files` is empty. `threads`: the query pool (and libsais's OpenMP threads with the sa-openmp feature).
    #[new]
    #[pyo3(signature = (files, threads, tokens=None, force_p40=false))]
    fn new(py: Python<'_>, files: Vec<String>, threads: usize, tokens: Option<PyReadonlyArray1<u16>>, force_p40: bool) -> PyResult<Self> {
        let given = match tokens {
            Some(t) if files.is_empty() => Some(t.as_slice()?.to_vec()),
            None => None,
            Some(_) => return Err(PyValueError::new_err("pass files or tokens, not both")),
        };
        let threads = threads.max(1);
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(threads)
            .thread_name(|i| format!("sa-count-{i}"))
            .build()
            .map_err(|e| PyValueError::new_err(e.to_string()))?;
        // the read and the sort without the GIL: the trainer's main thread runs on while a side thread builds
        let core = py.detach(|| -> PyResult<Core> {
            let text = match given {
                Some(t) => t,
                None => read_share(&files)?,
            };
            pool.install(|| Core::build(text, threads, force_p40)).map_err(PyValueError::new_err)
        })?;
        Ok(Self { core, pool })
    }

    /// ExactCounts' counts of this share at every query into out [n, 14] (int32): N for context lengths 2..8, then C.
    fn counts_into(&self, py: Python<'_>, tok: PyReadonlyArray1<i32>, tgt: PyReadonlyArray1<i32>, start: PyReadonlyArray1<i64>,
                   mut out: PyReadwriteArray2<i32>) -> PyResult<()> {
        let (tok, tgt, start) = (tok.as_slice()?, tgt.as_slice()?, start.as_slice()?);
        let out = out.as_slice_mut()?;
        py.detach(|| self.pool.install(|| self.core.counts(tok, tgt, start, out))).map_err(PyValueError::new_err)
    }

    /// The infini-gram backoff per query: (length, count) of the longest context suffix that occurs, up to max_len.
    fn longest_into(&self, py: Python<'_>, tok: PyReadonlyArray1<i32>, start: PyReadonlyArray1<i64>, max_len: usize,
                    mut len: PyReadwriteArray2<i32>, mut cnt: PyReadwriteArray2<i64>) -> PyResult<()> {
        let (tok, start) = (tok.as_slice()?, start.as_slice()?);
        let (len, cnt) = (len.as_slice_mut()?, cnt.as_slice_mut()?);
        if len.len() != tok.len() || cnt.len() != tok.len() || start.len() != tok.len() {
            return Err(PyValueError::new_err("longest_into: tok, start, len and cnt must have one entry per query"));
        }
        if tok.iter().any(|t| !(0..SYMS as i32).contains(t)) {
            return Err(PyValueError::new_err("longest_into: tokens must be 0..65535"));
        }
        py.detach(|| self.pool.install(|| self.core.longest(tok, start, max_len, len, cnt)));
        Ok(())
    }

    /// The occurrences of one gram (reading order).
    fn count(&self, gram: Vec<i32>) -> PyResult<u64> {
        if gram.iter().any(|t| !(0..SYMS as i32).contains(t)) {
            return Err(PyValueError::new_err("tokens must be 0..65535"));
        }
        Ok(self.core.count(&gram))
    }

    #[getter]
    fn tokens(&self) -> usize {
        self.core.text.len()
    }

    /// Bytes held: the reversed text, the suffix array and the bucket table.
    #[getter]
    fn memory(&self) -> usize {
        self.core.memory()
    }

    #[getter]
    fn packed(&self) -> bool {
        matches!(self.core.sa, Sa::P40(_))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    /// A small deterministic generator (no rand dependency).
    struct Rng(u64);
    impl Rng {
        fn next(&mut self) -> u64 {
            self.0 ^= self.0 << 13;
            self.0 ^= self.0 >> 7;
            self.0 ^= self.0 << 17;
            self.0
        }
        fn below(&mut self, n: u64) -> u64 {
            self.next() % n
        }
    }

    const BOS: i32 = 50256;

    /// exact_counts.segment_starts: the latest of the query's chunk start and the last BOS at or before it.
    fn segment_starts(tok: &[i32], chunk: &[i64]) -> Vec<i64> {
        let mut s = -1i64;
        (0..tok.len()).map(|i| {
            s = s.max(chunk[i]).max(if tok[i] == BOS { i as i64 } else { 0 });
            s
        }).collect()
    }

    /// ExactCounts._count with exact keys for the hashes: the query table from _ins_kernel, the scan of every share
    /// (a gram counts only while its shorter suffix is in the table, as `alive = found`), _find_kernel's lookups; summed
    /// over the shares as the all-reduce does.
    fn reference(shares: &[Vec<u16>], tok: &[i32], tgt: &[i32], start: &[i64]) -> Vec<i32> {
        let n = tok.len();
        let mut table: HashMap<Vec<i32>, i64> = HashMap::new();
        for j in 0..n {
            for m in 1..=MAXO {
                if (j as i64) - (m as i64 - 1) < start[j] {
                    break;
                }
                let mut g = tok[j + 1 - m..=j].to_vec();
                g.push(tgt[j]);
                table.insert(g, 0);
            }
        }
        for share in shares {
            for p in 0..share.len() {
                for l in 2..=MAXO + 1 {
                    if p + 1 < l {
                        break;
                    }
                    let g: Vec<i32> = share[p + 1 - l..=p].iter().map(|&t| t as i32).collect();
                    match table.get_mut(&g) {
                        Some(c) => *c += 1,
                        None => break,
                    }
                }
            }
        }
        let mut out = vec![0i32; n * COLS];
        for i in 0..n {
            for m in 2..=MAXO {
                if (i as i64) - (m as i64 - 1) < start[i] {
                    break;
                }
                let ctx = tok[i + 1 - m..=i].to_vec();
                out[i * COLS + m - 2] = *table.get(&ctx).unwrap_or(&0) as i32;
                let mut g = ctx;
                g.push(tgt[i]);
                out[i * COLS + MAXO + m - 3] = *table.get(&g).unwrap_or(&0) as i32;
            }
        }
        out
    }

    fn sa_counts(shares: &[Vec<u16>], tok: &[i32], tgt: &[i32], start: &[i64], force_p40: bool) -> Vec<i32> {
        let mut total = vec![0i32; tok.len() * COLS];
        for share in shares {
            let core = Core::build(share.clone(), 1, force_p40).unwrap();
            let mut out = vec![0i32; tok.len() * COLS];
            core.counts(tok, tgt, start, &mut out).unwrap();
            total.iter_mut().zip(&out).for_each(|(t, o)| *t += o);
        }
        total
    }

    /// A corpus of documents (BOS-led) over a small vocabulary with repeated phrases, so long grams recur.
    fn corpus(rng: &mut Rng, n: usize, vocab: u64) -> Vec<u16> {
        let phrases: Vec<Vec<u16>> = (0..20).map(|_| (0..2 + rng.below(10)).map(|_| rng.below(vocab) as u16).collect()).collect();
        let mut v = Vec::with_capacity(n + 16);
        while v.len() < n {
            match rng.below(10) {
                0 => v.push(BOS as u16),
                1..=4 => v.extend(&phrases[rng.below(20) as usize]),
                _ => v.push(rng.below(vocab) as u16),
            }
        }
        v.truncate(n);
        v
    }

    /// Queries as the trainer makes them: a stream (target = next token) cut into chunks, and optionally broken
    /// (target != next token) and with the queries' contexts drawn from the corpus.
    fn queries(rng: &mut Rng, src: &[u16], n: usize, chunk: usize, broken: bool) -> (Vec<i32>, Vec<i32>, Vec<i64>) {
        let at = rng.below((src.len() - n - 1) as u64) as usize;
        let stream: Vec<i32> = src[at..at + n + 1].iter().map(|&t| t as i32).collect();
        let tok = stream[..n].to_vec();
        let mut tgt = stream[1..].to_vec();
        if broken {
            for _ in 0..n / 7 {
                let i = rng.below(n as u64) as usize;
                tgt[i] = src[rng.below(src.len() as u64) as usize] as i32;
            }
        }
        let ch: Vec<i64> = (0..n).map(|i| (i / chunk * chunk) as i64).collect();
        let start = segment_starts(&tok, &ch);
        (tok, tgt, start)
    }

    #[test]
    fn counts_match_reference_random() {
        let mut rng = Rng(0x9E3779B97F4A7C15);
        for (round, &(vocab, broken)) in [(5u64, false), (5, true), (40, false), (40, true), (50257, false), (300, true)].iter().enumerate() {
            let full = corpus(&mut rng, 60_000, vocab);
            let shares: Vec<Vec<u16>> = full.chunks(7_500).map(|c| c.to_vec()).collect(); // 8 shares
            let (tok, tgt, start) = queries(&mut rng, &full, 3_000, 512, broken);
            let want = reference(&shares, &tok, &tgt, &start);
            assert_eq!(sa_counts(&shares, &tok, &tgt, &start, round % 2 == 1), want, "round {round} vocab {vocab} broken {broken}");
            assert!(want.iter().filter(|&&c| c > 0).count() > 1000, "round {round}: too few nonzero counts to mean much");
        }
    }

    #[test]
    fn queries_from_other_text_and_unseen_tokens() {
        let mut rng = Rng(12345);
        let shares: Vec<Vec<u16>> = (0..3).map(|_| corpus(&mut rng, 5_000, 30)).collect();
        let other = corpus(&mut rng, 4_000, 33); // tokens 30..32 never occur in the shares
        let (tok, tgt, start) = queries(&mut rng, &other, 2_000, 300, true);
        assert_eq!(sa_counts(&shares, &tok, &tgt, &start, false), reference(&shares, &tok, &tgt, &start));
    }

    #[test]
    fn uninserted_context_counts_zero() {
        // query 1's 2-token context [1, 2] occurs in the corpus, but query 0's target is 9, not 2, so nothing inserts it
        let shares = vec![vec![1u16, 2, 3, 1, 2, 3]];
        let (tok, tgt) = (vec![1, 2], vec![9, 3]);
        let start = vec![0i64, 0];
        let want = reference(&shares, &tok, &tgt, &start);
        assert_eq!(want[COLS], 0);
        assert_eq!(sa_counts(&shares, &tok, &tgt, &start, false), want);
        assert_eq!(Core::build(shares[0].clone(), 1, false).unwrap().count(&[1, 2]), 2);
        // and inserted by a later query that ends with it: counted
        let (tok, tgt) = (vec![1, 2, 1], vec![9, 3, 2]);
        let start = vec![0i64, 0, 0];
        let want = reference(&shares, &tok, &tgt, &start);
        assert_eq!(want[COLS], 2);
        assert_eq!(sa_counts(&shares, &tok, &tgt, &start, false), want);
    }

    #[test]
    fn grams_do_not_cross_shares_but_cross_files() {
        // [7, 8] spans the two shares' junction (not counted) and a file junction inside share 1 (counted)
        let shares = vec![vec![5u16, 7], vec![8, 5, 7, 8]];
        let (tok, tgt, start) = (vec![5, 7], vec![7, 8], vec![0i64, 0]);
        let got = sa_counts(&shares, &tok, &tgt, &start, false);
        assert_eq!(got, reference(&shares, &tok, &tgt, &start));
        assert_eq!(got[COLS + MAXO - 1], 1); // C of [5, 7] + 8: once, in share 1
    }

    #[test]
    fn longest_match() {
        let text: Vec<u16> = vec![1, 2, 3, 4, 1, 2, 3, 5, 2, 3, 4];
        let core = Core::build(text.clone(), 1, true).unwrap();
        let tok = vec![9, 1, 2, 3, 4];
        let start = vec![0i64; 5];
        let (mut len, mut cnt) = (vec![0i32; 5], vec![0i64; 5]);
        core.longest(&tok, &start, 100, &mut len, &mut cnt);
        assert_eq!(len, vec![0, 1, 2, 3, 4]); // [9] never occurs; [1,2,3,4] once
        assert_eq!(cnt, vec![11, 2, 2, 2, 1]);
        let start = vec![0i64, 0, 0, 2, 2];
        core.longest(&tok, &start, 100, &mut len, &mut cnt);
        assert_eq!(len[4], 3); // [2,3,4] from the segment start at 2
        assert_eq!(cnt[4], 2);
    }

    #[test]
    fn count_matches_brute_force() {
        let mut rng = Rng(777);
        let text = corpus(&mut rng, 20_000, 12);
        for force in [false, true] {
            let core = Core::build(text.clone(), 1, force).unwrap();
            assert_eq!(core.packed_is(force), true);
            for _ in 0..300 {
                let l = 1 + rng.below(9) as usize;
                let at = rng.below((text.len() - l) as u64) as usize;
                let gram: Vec<i32> = text[at..at + l].iter().map(|&t| t as i32).collect();
                let brute = text.windows(l).filter(|w| w.iter().zip(&gram).all(|(&a, &b)| a as i32 == b)).count() as u64;
                assert_eq!(core.count(&gram), brute);
            }
        }
    }

    impl Core {
        fn packed_is(&self, p: bool) -> bool {
            matches!(self.sa, Sa::P40(_)) == p
        }
    }
}
