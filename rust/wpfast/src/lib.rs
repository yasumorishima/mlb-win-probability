//! Rust versions of two frozen Python routines of the player layer.
//! The Python files stay the reference (research/player_layer/layer.py,
//! game.py / game_pen.py); tests/test_fast_bench.py compares outputs.
//! Each function receives whole tables in one call (no per-row crossing).

use numpy::{IntoPyArray, PyArray1, PyReadonlyArray2, PyReadonlyArray4};
use pyo3::exceptions::PyValueError;
use pyo3::prelude::*;

/// layer.run_dist: P (NS, C+1), T (C+1, MAXR+1, NS, NS+1), M (9, C)
/// -> flat (9 * NS * (max_runs + 1)), index [j][s][k].
#[pyfunction]
fn run_dist<'py>(
    py: Python<'py>,
    p: PyReadonlyArray2<'py, f64>,
    t: PyReadonlyArray4<'py, f64>,
    m: PyReadonlyArray2<'py, f64>,
    max_runs: usize,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let p = p.as_array();
    let t = t.as_array();
    let m = m.as_array();
    let ns = p.shape()[0];
    let c = p.shape()[1] - 1;
    let ts = t.shape();
    let maxr = ts[1] - 1;
    if ts[0] != c + 1 || ts[2] != ns || ts[3] != ns + 1 || m.shape() != [9, c] {
        return Err(PyValueError::new_err("run_dist: shape mismatch"));
    }
    let n = 9 * ns;
    // q[j][s][ci]
    let mut q = vec![0.0f64; 9 * ns * c];
    for j in 0..9 {
        for s in 0..ns {
            let po = p[[s, c]];
            let mut tot = 0.0;
            for ci in 0..c {
                let v = p[[s, ci]] * m[[j, ci]];
                q[(j * ns + s) * c + ci] = v;
                tot += v;
            }
            for ci in 0..c {
                let x = &mut q[(j * ns + s) * c + ci];
                *x = *x / tot * (1.0 - po);
            }
        }
    }
    let mut mr = vec![0.0f64; (maxr + 1) * n * n];
    let mut a = vec![0.0f64; (maxr + 1) * n];
    for j in 0..9 {
        let jn = (j + 1) % 9;
        for ci in 0..=c {
            for si in 0..ns {
                let w = if ci < c { q[(j * ns + si) * c + ci] } else { p[[si, c]] };
                let row = j * ns + si;
                let col0 = if ci < c { jn * ns } else { j * ns };
                for r in 0..=maxr {
                    for tt in 0..ns {
                        mr[(r * n + row) * n + col0 + tt] += w * t[[ci, r, si, tt]];
                    }
                    a[r * n + row] += w * t[[ci, r, si, ns]];
                }
            }
        }
    }
    // sparse copy of Mr[r] rows for r >= 1 (each row reaches at most two slot blocks)
    let mut sp: Vec<Vec<(usize, f64)>> = Vec::with_capacity((maxr + 1) * n);
    for r in 0..=maxr {
        for i in 0..n {
            let row = &mr[(r * n + i) * n..(r * n + i + 1) * n];
            sp.push(row.iter().enumerate().filter(|(_, v)| **v != 0.0).map(|(k, v)| (k, *v)).collect());
        }
    }
    let out = py.allow_threads(|| {
        // LU with partial pivoting of I - Mr[0]
        let mut lu = vec![0.0f64; n * n];
        for i in 0..n {
            for k in 0..n {
                lu[i * n + k] = if i == k { 1.0 } else { 0.0 } - mr[i * n + k];
            }
        }
        let mut piv: Vec<usize> = (0..n).collect();
        for k in 0..n {
            let mut best = k;
            let mut bv = lu[k * n + k].abs();
            for i in k + 1..n {
                let v = lu[i * n + k].abs();
                if v > bv { bv = v; best = i; }
            }
            if best != k {
                for x in 0..n { lu.swap(k * n + x, best * n + x); }
                piv.swap(k, best);
            }
            let (top, rest) = lu.split_at_mut((k + 1) * n);
            let pivrow = &top[k * n..(k + 1) * n];
            let d = pivrow[k];
            for i in 0..(n - k - 1) {
                let row = &mut rest[i * n..(i + 1) * n];
                let f = row[k] / d;
                row[k] = f;
                if f != 0.0 {
                    for (x, y) in row[k + 1..].iter_mut().zip(&pivrow[k + 1..]) { *x -= f * y; }
                }
            }
        }
        let nr = max_runs + 1;
        let mut out = vec![0.0f64; nr * n];
        let mut rhs = vec![0.0f64; n];
        let mut y = vec![0.0f64; n];
        for k in 0..nr {
            for i in 0..n { rhs[i] = if k <= maxr { a[k * n + i] } else { 0.0 }; }
            for r in 1..=maxr {
                if k >= r {
                    let prev = &out[(k - r) * n..(k - r + 1) * n];
                    for i in 0..n {
                        let mut s = 0.0;
                        for &(x, v) in &sp[r * n + i] { s += v * prev[x]; }
                        rhs[i] += s;
                    }
                }
            }
            for i in 0..n {
                let mut s = rhs[piv[i]];
                let row = &lu[i * n..i * n + i];
                for (a1, b1) in row.iter().zip(&y[..i]) { s -= a1 * b1; }
                y[i] = s;
            }
            let (done, _) = out.split_at_mut((k + 1) * n);
            let ok = &mut done[k * n..];
            for i in (0..n).rev() {
                let mut s = y[i];
                let row = &lu[i * n + i + 1..(i + 1) * n];
                for (a1, b1) in row.iter().zip(&ok[i + 1..]) { s -= a1 * b1; }
                ok[i] = s / lu[i * n + i];
            }
        }
        // transpose to [j][s][k] = out[k][j*ns+s]
        let mut res = vec![0.0f64; n * nr];
        for i in 0..n { for k in 0..nr { res[i * nr + k] = out[k * n + i]; } }
        res
    });
    Ok(out.into_pyarray(py))
}

/// game.game_wp / game_pen.game_wp with every start distribution given as a
/// table: dist (max_inning, 2, ND, R) for innings 1..max_inning, half 0 = top,
/// 1 = bottom (a difference-free distribution is the same row repeated ND
/// times, which gives the same sums as the 1-D branch).
/// Returns flat (max_inning + 1, 2, ND): H at the start of inning i (1-based).
#[pyfunction]
fn game_wp<'py>(
    py: Python<'py>,
    dist: PyReadonlyArray4<'py, f64>,
    sched: i64,
    max_diff: i64,
) -> PyResult<Bound<'py, PyArray1<f64>>> {
    let d = dist.as_array();
    let sh = d.shape();
    let (mi, nd, nr) = (sh[0], sh[2], sh[3]);
    if sh[1] != 2 || nd as i64 != 2 * max_diff + 1 {
        return Err(PyValueError::new_err("game_wp: shape mismatch"));
    }
    let dv: Vec<i64> = (0..nd as i64).map(|i| i - max_diff).collect();
    let clip = |x: i64| -> usize { (x.clamp(-max_diff, max_diff) + max_diff) as usize };
    // h[(inn-1)*2 + half][di], inn = 1..=mi+1
    let mut h = vec![0.0f64; (mi + 1) * 2 * nd];
    for half in 0..2 {
        for i in 0..nd {
            let s = (dv[i] as f64).signum() * if dv[i] == 0 { 0.0 } else { 1.0 };
            h[(mi * 2 + half) * nd + i] = 0.5 + 0.04 * s;
        }
    }
    let mut v = vec![0.0f64; nd];
    let mut total = vec![0.0f64; nd];
    for inn in (1..=mi).rev() {
        for &half in &[1usize, 0usize] {
            for x in total.iter_mut() { *x = 0.0; }
            let ii = inn as i64;
            for r in 0..nr {
                let any = (0..nd).any(|i| d[[inn - 1, half, i, r]] != 0.0);
                if !any { continue; }
                if half == 0 {
                    let nxt = (inn - 1) * 2 + 1;                 // bottom of the same inning
                    for i in 0..nd {
                        let di = clip(dv[i] - r as i64);
                        let mut val = h[nxt * nd + di];
                        if ii >= sched && dv[di] > 0 { val = 1.0; }
                        v[i] = val;
                    }
                } else {
                    let nxt = inn * 2;                             // top of the next inning
                    for i in 0..nd {
                        let di = clip(dv[i] + r as i64);
                        v[i] = if ii >= sched {
                            let d2 = dv[di];
                            if d2 > 0 { 1.0 } else if d2 == 0 { h[nxt * nd + max_diff as usize] } else { 0.0 }
                        } else { h[nxt * nd + di] };
                    }
                }
                for i in 0..nd { total[i] += d[[inn - 1, half, i, r]] * v[i]; }
            }
            let base = ((inn - 1) * 2 + half) * nd;
            h[base..base + nd].copy_from_slice(&total);
        }
    }
    Ok(h.into_pyarray(py))
}

/// FNV-1a 64 hash of the src/lib.rs this binary was built from (build.rs).
#[pyfunction]
fn source_fnv() -> &'static str {
    env!("WPFAST_SRC_FNV")
}

#[pymodule]
fn wpfast(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(source_fnv, m)?)?;
    m.add_function(wrap_pyfunction!(run_dist, m)?)?;
    m.add_function(wrap_pyfunction!(game_wp, m)?)?;
    Ok(())
}
