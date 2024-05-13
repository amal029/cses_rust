// XXX: This file does many algorithms from CSES website.
// Website: https://cses.fi/problemset/list/

use std::cmp::{max, min, Reverse};
use std::collections::{BinaryHeap, HashMap, VecDeque};
use std::iter::zip;
use std::process::exit;
use std::ptr::swap;
use std::usize::MAX;

type Us = usize;

// XXX: Just made a generic reader for all problems
fn _read<T>() -> Vec<T>
where
    T: std::str::FromStr,
{
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    line.trim()
        .split(' ')
        .flat_map(str::parse::<T>)
        .collect::<Vec<_>>()
}

fn _weird_algo() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<i32>)
        .collect::<Vec<_>>();
    let mut n: i32 = nums[0];
    print!("{} ", n);
    while n != 1 {
        if n % 2 == 0 {
            n /= 2;
        } else {
            n *= 3;
            n += 1;
        }
        print!("{} ", n);
    }
    println!();
}

fn _two_sets() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<i32>)
        .collect::<Vec<_>>();
    let mut n: i32 = nums[0];
    let on = n as usize;
    // XXX: If n*(n+1)/2 % 2 == 0 --> it is divisible.
    let sum = n * (n + 1) / 2;
    if sum % 2 == 0 {
        println!("YES");
        let y = sum / 2;
        let mut v1 = vec![];
        let mut ss: i32 = v1.iter().sum();
        while ss != y {
            if ss + n <= y {
                v1.push(n);
                n -= 1;
            } else {
                v1.push(y - ss);
            }
            ss = v1.iter().sum();
        }
        println!("{}", v1.len());
        v1.iter().for_each(|x| print!("{} ", x));
        println!("\n{}", (on - v1.len()));
        // XXX: This borrows the value of x instead of copying!
        let start = match v1.iter().min() {
            Some(x) => x,
            None => exit(1),
        };
        for i in *start + 1..(n + 1) {
            print!("{} ", i);
        }
        println!();
    } else {
        println!("NO");
    }
}

// XXX: This is checking if the graph is bi-partite.
fn _building_teams() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<usize>)
        .collect::<Vec<_>>();
    let n: usize = nums[0];
    let m: usize = nums[1];
    let mut vs: Vec<Vec<usize>> = vec![vec![]; n];
    vs.reserve(n);
    let mut counter = 0;
    while counter < m {
        line.clear();
        std::io::stdin().read_line(&mut line).expect("input");
        let nums = line
            .trim()
            .split(' ')
            .flat_map(str::parse::<usize>)
            .collect::<Vec<_>>();

        let mut s: usize = nums[0];
        s -= 1;
        let mut e: usize = nums[1];
        e -= 1;
        vs[s].push(e);
        vs[e].push(s);
        counter += 1;
    }
    // XXX: Now color the vertices with 1 or 2
    let mut colors: Vec<usize> = vec![0; n];
    fn dfs_color(i: usize, vs: &Vec<Vec<usize>>, crs: &mut Vec<usize>) -> bool {
        // println!("Doing : {i}");
        // println!("{:?}", crs);
        let mut cc = vec![];
        // XXX: All loop borrow via reference
        for c in vs[i].iter() {
            // println!("{:?} for {:?}", crs[*c], *c);
            cc.push(crs[*c]);
        }
        if !cc.iter().all(|&x| x == 0 || x == 1 || x == 2) {
            return false;
        } else if cc.iter().any(|&x| x == 1) {
            // println!("giving color 2");
            crs[i] = 2;
        } else if cc.iter().any(|&x| x == 2 || x == 0) {
            // println!("giving color 1");
            crs[i] = 1;
        } // else if cc.iter().all(|&x| x == 0) {
          //     // println!("giving color 1");
          //     crs[i] = 1;
          // }
          // XXX: Now do this for its neighbors.
        let mut dd = true;
        for c1 in vs[i].iter() {
            if crs[*c1] == 0 && !dfs_color(*c1, vs, crs) {
                dd = false;
                break;
            }
        }
        dd
    }
    // XXX: Do this for all nodes
    for nodes in 0..n {
        if !dfs_color(nodes, &vs, &mut colors) {
            println!("IMPOSSIBLE");
            break;
        }
    }
    for c in colors.iter() {
        print!("{c} ");
    }
    println!();
}

fn _course_schedule() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<usize>)
        .collect::<Vec<_>>();
    let n = nums[0];
    let m = nums[1];
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    let mut counter = 0;
    while counter < m {
        line.clear();
        std::io::stdin().read_line(&mut line).expect("input");
        let nums = line
            .trim()
            .split(' ')
            .flat_map(str::parse::<usize>)
            .collect::<Vec<_>>();
        let mut s: usize = nums[0];
        let mut e: usize = nums[1];
        s -= 1;
        e -= 1;
        // XXX: This is a directed graph
        adj[s].push(e);
        counter += 1;
    }
    // XXX: Do a topological sort of the DAG, we need to check that the
    // graph is acyclic while traversing it.
    let mut order = Vec::with_capacity(n);
    // XXX: Note that piece of junk that is rust does not do
    // vector<bool> optimisation like C++! it allocates byte for each
    // bool -- 7 wasted bits/bool! However, for now I am just going to
    // use vector<bool> for memoization.
    let mut vis = vec![false; n];
    fn top_sort(
        adj: &Vec<Vec<usize>>,
        order: &mut Vec<usize>,
        vis: &mut Vec<bool>,
        i: usize,
    ) -> bool {
        vis[i] = true;
        // XXX: Now do a post-order traversal of the graph
        let mut dd = true;
        for c in adj[i].iter() {
            if !vis[*c] {
                if !top_sort(adj, order, vis, *c) {
                    dd = false;
                    break;
                }
            } else {
                // XXX: Check that if it is already visited then it
                // should be in the order vector.
                match order.iter().find(|&x| x == c) {
                    Some(_) => (),
                    None => return false,
                }
            }
        }
        if dd {
            order.push(i);
        }
        dd
    }
    for c in 0..n {
        if !vis[c] && !top_sort(&adj, &mut order, &mut vis, c) {
            println!("IMPOSSIBLE");
            return;
        }
    }

    for c in order.iter().rev() {
        print!("{} ", c + 1);
    }
    println!();
}

fn _longest_flight_route() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<usize>)
        .collect::<Vec<_>>();
    let n: usize = nums[0];
    let m: usize = nums[1];
    let mut counter = 0;
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    while counter < m {
        line.clear();
        std::io::stdin().read_line(&mut line).expect("input");
        let nums = line
            .trim()
            .split(' ')
            .flat_map(str::parse::<usize>)
            .collect::<Vec<_>>();
        let mut s = nums[0];
        s -= 1;
        let mut e = nums[1];
        e -= 1;
        // XXX: A DAG
        adj[s].push(e);
        counter += 1;
    }
    // XXX: Note that piece of junk that is rust does not do
    // vector<bool> optimisation like C++! it allocates byte for each
    // bool -- 7 wasted bits/bool! However, for now I am just going to
    // use vector<bool> for memoization.
    let mut vis: Vec<bool> = vec![false; n];
    let mut paths: Vec<Vec<usize>> = vec![vec![]; n];

    // XXX: Now just do DP for finding the longest path via DFS
    fn _get_path(
        i: usize,
        vis: &mut Vec<bool>,
        paths: &mut Vec<Vec<usize>>,
        adj: &Vec<Vec<usize>>,
        dest: usize,
    ) {
        vis[i] = true;
        // XXX: The base case -- this is the destination
        if i == dest {
            paths[i].push(dest);
            return;
        }
        // XXX: Recursive case
        for c in adj[i].iter() {
            if !vis[*c] {
                _get_path(*c, vis, paths, adj, dest);
            }
        }
        // XXX: Get the max path from all the children paths
        let mut mm = 0usize;
        let mut cmi = -1i64;
        for c in adj[i].iter() {
            (cmi, mm) = if paths[*c].len() > mm {
                (*c as i64, paths[*c].len())
            } else {
                (cmi, mm)
            };
        }
        // XXX: Add your path XXX: copy the longest path to yourself
        // from cmi -- this can be made much better -- just hold pointer
        // (int) to next node in longest path!
        if cmi != -1 {
            paths[i].push(i);
            let tt = paths[cmi as usize].clone();
            paths[i].extend(tt);
        }
    }

    // XXX: Call dp for all node if they are already not visited
    for i in 0..n {
        if !vis[i] {
            _get_path(i, &mut vis, &mut paths, &adj, n - 1);
        }
    }
    println!("{}", paths[0].len());
    for i in paths[0].iter() {
        print!("{} ", i + 1);
    }
    println!();
}

fn _planets_qs1() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<usize>)
        .collect::<Vec<_>>();
    let n = nums[0];
    let q = nums[1];
    line.clear();
    std::io::stdin().read_line(&mut line).expect("input");
    let mut nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<usize>)
        .collect::<Vec<_>>();
    let mut dests: Vec<usize> = vec![0; n];
    for (ii, uu) in nums.iter_mut().enumerate() {
        *uu -= 1;
        dests[ii] = *uu;
    }
    // XXX: Now read the queries
    let mut qs: Vec<(usize, usize)> = vec![(0, 0); q];
    for cc in qs.iter_mut().take(q) {
        line.clear();
        std::io::stdin().read_line(&mut line).expect("input");
        let nums = line
            .trim()
            .split(' ')
            .flat_map(str::parse::<usize>)
            .collect::<Vec<_>>();
        *cc = (nums[0] - 1, nums[1]);
    }
    // XXX: Now just traverse the vector depending upon the query
    for (s, k) in qs.iter_mut() {
        while *k > 0 {
            *s = dests[*s];
            *k -= 1;
        }
        println!("{}", *s + 1);
    }
}

fn _game_routes() {
    let nums = _read::<usize>();
    let vs = nums[0];
    let es = nums[1];
    let mut adj: Vec<Vec<usize>> = vec![vec![]; vs];
    for _ in 0..es {
        let ss = _read::<usize>();
        adj[ss[0] - 1].push(ss[1] - 1);
    }
    // XXX: Now just do dp to find the total number of paths from source
    // (0) to destination (n-1)
    let mut paths = vec![0usize; vs];
    fn dp_path(adj: &Vec<Vec<usize>>, paths: &mut Vec<usize>, dest: usize, s: usize) {
        if s == dest {
            paths[s] += 1;
        }
        for c in adj[s].iter() {
            if paths[*c] == 0 {
                dp_path(adj, paths, dest, *c);
            }
            paths[s] += paths[*c];
        }
    }
    dp_path(&adj, &mut paths, vs - 1, 0);
    println!("{}", paths[0]);
}

fn _road_reparation() {
    let ss = _read::<usize>();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; ss[0]];
    let mut dd: Vec<bool> = vec![false; ss[0]];
    let mut pq = BinaryHeap::with_capacity(ss[1]);
    for _ in 0..ss[1] {
        let ss = _read::<usize>();
        // XXX: Cost, source node, destination node -- reverse for
        // min-heap
        pq.push(Reverse((ss[2], ss[0] - 1, ss[1] - 1)));
    }
    // XXX: Now just go through the pq and put the least cost edges in
    // adj list
    let mut total_cost = 0;
    while !pq.is_empty() {
        let Reverse((c, s, d)) = pq.pop().unwrap();
        // XXX: If the source and destination are both done then leave
        // the edge
        if !dd[s] || !dd[d] {
            adj[s].push(d);
            adj[d].push(s);
            dd[s] = true;
            dd[d] = true;
            total_cost += c;
        }
    }
    // XXX: Do a dfs from any starting node. We should be able to reach
    // every node from the starting node.
    let mut vis = vec![false; ss[0]];
    fn reachable(adj: &Vec<Vec<usize>>, vis: &mut Vec<bool>, s: usize) {
        vis[s] = true;
        for c in adj[s].iter() {
            if !vis[*c] {
                reachable(adj, vis, *c);
            }
        }
    }
    reachable(&adj, &mut vis, 0);
    if !vis.iter().all(|&x| x) {
        println!("IMPOSSIBLE");
    } else {
        println!("{}", total_cost);
    }
}

fn _kosaraju(adj: &Vec<Vec<usize>>, s: usize, vis: &mut Vec<bool>, paths: &mut Vec<usize>) {
    vis[s] = true;
    for c in adj[s].iter() {
        if !vis[*c] {
            _kosaraju(adj, *c, vis, paths);
        }
    }
    // XXX: Now add yourself to the paths -- top sorted already
    paths.push(s);
}

fn _flight_routes_check() {
    let ss = _read::<usize>();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; ss[0]];
    let mut iadj: Vec<Vec<usize>> = vec![vec![]; ss[0]];
    for _ in 0..ss[1] {
        let ss = _read::<usize>();
        // This is a DAG
        adj[ss[0] - 1].push(ss[1] - 1);
        // XXX: The inverted DAG
        iadj[ss[1] - 1].push(ss[0] - 1);
    }
    // TODO: Just do Kosaraju' algo
    let mut paths: Vec<usize> = vec![];
    let mut vis = vec![false; ss[0]];
    _kosaraju(&adj, 0, &mut vis, &mut paths);
    // XXX: Now go in the opposite direction using iadj
    vis.fill(false);
    let mut sccs: Vec<Vec<usize>> = vec![];
    for c in paths.iter().rev() {
        let mut scc = vec![];
        if !vis[*c] {
            _kosaraju(&iadj, *c, &mut vis, &mut scc);
        }
        if !scc.is_empty() {
            sccs.push(scc);
        }
    }
    // XXX: Now if the length of sccs is > 1, then some flights cannot
    // reach some destination.
    if sccs.len() > 1 {
        println!("NO");
        println!("{} {}", sccs[0][0] + 1, sccs[1][0] + 1);
    }
}

fn _teleporters_path() {
    let ss = _read::<usize>();
    let mut degs: Vec<usize> = vec![0; ss[0]];
    let mut adj: Vec<Vec<(usize, bool)>> = vec![vec![]; ss[0]];
    for _ in 0..ss[1] {
        let u = _read::<usize>();
        // XXX: true here means that the edge is still connected
        adj[u[0] - 1].push((u[1] - 1, true));
        degs[u[0] - 1] += 1;
        degs[u[1] - 1] += 1;
    }
    if !degs.iter().all(|&x| x % 2 == 0) && degs.iter().filter(|&x| x % 2 != 0).count() != 2 {
        println!("IMPOSSIBLE");
        return;
    }
    // XXX: Now just do Eulerian path
    let mut q = vec![0];
    let mut path = Vec::with_capacity(ss[0]);
    // XXX: DFS search for eulerain path
    while !q.is_empty() {
        // XXX: It still does bound checking, even in unsafe block
        let dq = &mut degs[*q.last().unwrap()];
        if *dq == 0 {
            path.push(q.pop());
        } else {
            for (c, b) in adj[*q.last().unwrap()].iter_mut() {
                if *b {
                    *b = false;
                    *dq -= 1;
                    degs[*c] -= 1;
                    q.push(*c);
                    break;
                }
            }
        }
    }
    for c in path.iter().rev() {
        print!("{} ", c.unwrap() + 1);
    }
    println!();
}

// XXX: This is an NP-hard problem, and I am not doing any memoization
// currently for this example, because it is pretty small.
fn _hamiltonian_flights() {
    let ss = _read::<usize>();
    let n = ss[0];
    let m = ss[1];
    // XXX: Since 20 is max for n, we are going to solve it using dfs
    // and backtracking.
    let mut adj: Vec<Vec<usize>> = vec![vec![]; n];
    for _ in 0..m {
        let ss = _read::<usize>();
        adj[ss[0] - 1].push(ss[1] - 1);
    }
    let mut vis = vec![false; n];
    fn dfs_hamiltonian(adj: &[Vec<usize>], vis: &mut [bool], s: usize, d: usize, paths: &mut i32) {
        vis[s] = true;
        if s == d {
            if vis.iter().all(|&x| x) {
                *paths += 1;
            }
            vis[s] = false;
            return;
        }
        for c in adj[s].iter() {
            if !vis[*c] {
                dfs_hamiltonian(adj, vis, *c, d, paths);
            }
        }
        // XXX: remove yourself from vis
        vis[s] = false;
    }
    let mut paths = 0;
    dfs_hamiltonian(&adj, &mut vis, 0, n - 1, &mut paths);
    println!("{}", paths);
}

fn _giant_pizza() {
    let ss = _read::<usize>();
    let n = ss[0]; //the number of family members
    let m = ss[1]; // the number of toppings
    let mut adj: Vec<Vec<usize>> = vec![vec![]; m * 2];
    let mut iadj: Vec<Vec<usize>> = vec![vec![]; m * 2];
    for _ in 0..n {
        let mut line = String::new();
        std::io::stdin().read_line(&mut line).expect("input");
        let ss = line.trim().split(' ').collect::<Vec<_>>();
        let f = str::parse::<usize>(ss[1]).unwrap() - 1;
        let s = str::parse::<usize>(ss[3]).unwrap() - 1;
        let (a, b, c, d) = match (ss[0], ss[2]) {
            ("+", "+") => (f + m, s, s + m, f),
            ("+", "-") => (f + m, s + m, s, f + m),
            ("-", "+") => (f, s, s + m, f + m),
            ("-", "-") => (f, s + m, s, f + m),
            _ => exit(1),
        };
        // XXX: The implication graph
        adj[a].push(b);
        adj[c].push(d);
        // XXX: The inverted implication graph
        iadj[b].push(a);
        iadj[d].push(c);
    }
    // println!("{:?}", adj);
    // println!("{:?}", iadj);
    // XXX: Now just do a top sort for the adj
    let mut vis = vec![false; m * 2];
    let mut paths = vec![];
    for c in 0..m * 2 {
        if !vis[c] {
            _kosaraju(&adj, c, &mut vis, &mut paths);
        }
    }
    vis.fill(false); //cleared the vector
                     // println!("paths: {:?}", paths);

    // XXX: Now get the scc
    let mut sccs: Vec<Vec<usize>> = vec![];
    for c in paths.iter().rev() {
        let mut scc = vec![];
        if !vis[*c] {
            _kosaraju(&iadj, *c, &mut vis, &mut scc);
        }
        if !scc.is_empty() {
            sccs.push(scc);
        }
    }
    // println!("{:?}", sccs);
    // xxx: Check that forall a, a /\ !a do not occur in the same scc
    if sccs.iter().all(|x| x.len() == 1) {
        let mut ans = vec!['a'; m];
        for c in sccs.iter() {
            if c[0] >= m {
                ans[c[0] - m] = if ans[c[0] - m] == 'a' {
                    '+'
                } else {
                    ans[c[0] - m]
                }
            } else {
                ans[c[0]] = if ans[c[0]] == 'a' { '-' } else { ans[c[0]] }
            }
        }
        ans.iter().for_each(|&x| print!("{} ", x));
        println!();
    } else {
        // TODO: do a reachability check for i in 0 .. m, can i reach
        // i+m in any scc?
        println!("IMPOSSIBLE");
    }
}

fn _g_match(m: &[Vec<Us>], cols: &mut [i64], tom: Us, row: Us, col: Us) -> bool {
    // XXX: Move through each row and get the first to_match.
    let cc = m[row]
        .iter()
        .enumerate()
        .filter(|(t, &x)| x == tom && *t >= col)
        .map(|(t, &_)| t)
        .collect::<Vec<usize>>();
    let mut done = false;
    // XXX: This gives better temporal/spatial locality
    for j in cc.iter() {
        if cols[*j] == -1 {
            let idx = cols
                .iter()
                .enumerate()
                .filter(|&(_, x)| *x as usize == row)
                .map(|(t, &_)| t)
                .collect::<Vec<usize>>();
            assert!(idx.len() <= 1);
            if idx.len() == 1 {
                cols[idx[0]] = -1
            } // remove yourself
            cols[*j] = row as i64; // add yourself to the new position
            done = true;
            break;
        }
    }
    if !done {
        // XXX: Now re-assign a previous col if you can
        for &_j in cc.iter() {
            // XXX: Try moving one by one if possible
            done = _g_match(m, cols, tom, cols[_j] as usize, _j + 1);
            if done {
                // XXX: Add yourself to the cols
                cols[_j] = row as i64;
                break;
            }
        }
    }
    done
}

// XXX: Maximum matching for a bi-partite graph
fn _school_dance() {
    let ss = _read::<usize>();
    let mut adj: Vec<Vec<usize>> = vec![vec![0; ss[1]]; ss[0]];
    for _ in 0..ss[2] {
        let ss = _read::<usize>();
        adj[ss[0] - 1][ss[1] - 1] = 1;
    }
    // XXX: Now just do a greedy matching algorithm
    let mut cols = vec![-1i64; ss[1]];
    for i in 0..ss[0] {
        _g_match(&adj, &mut cols, 1, i, 0);
    }
    let cc = cols
        .iter()
        .enumerate()
        .filter(|&(_, x)| *x != -1)
        .map(|(t, &x)| (t, x))
        .collect::<Vec<(usize, i64)>>();
    println!("{}", cc.len());
    cc.iter().for_each(|&(t, x)| {
        println!("{} {}", x + 1, t + 1);
    });
}

fn _bfs_max_flow(
    s: usize,
    t: usize,
    ps: &mut [usize],
    adj: &[Vec<usize>],
    cap: &[Vec<i64>],
) -> i64 {
    // XXX: Make a queue
    ps.fill(MAX);
    ps[s] = MAX - 1; //just to be different
    let mut q = VecDeque::new();
    q.push_back((s, MAX));
    let mut flow = 0;
    while !q.is_empty() {
        let (p, v) = q.pop_front().unwrap();
        if p == t {
            // XXX: We reached the terminal node
            flow = v;
            break;
        }
        for &c in adj[p].iter() {
            let cv = cap[p][c] as usize;
            if ps[c] == MAX && cv > 0 {
                q.push_back((c, std::cmp::min(v, cv)));
                ps[c] = p;
            }
        }
    }
    flow as i64
}

fn _police_chase() {
    const VAL: i64 = 1;
    let ss = _read::<usize>();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; ss[0]];
    let mut cap: Vec<Vec<i64>> = vec![vec![-1; ss[0]]; ss[0]];
    for _ in 0..ss[1] {
        let s = _read::<usize>();
        adj[s[0] - 1].push(s[1] - 1);
        adj[s[1] - 1].push(s[0] - 1);
        cap[s[0] - 1][s[1] - 1] = VAL;
        cap[s[1] - 1][s[0] - 1] = 0;
    }

    // XXX: Now just do ford-fulkerson algorithm
    let mut flow: i64;
    let mut parents: Vec<usize> = vec![MAX; ss[0]];
    loop {
        flow = _bfs_max_flow(0, ss[0] - 1, &mut parents, &adj, &cap);
        if flow == 0 {
            break;
        };
        // XXX: Update the capacity matrix
        let mut c: usize = ss[0] - 1;
        let mut p: usize = parents[c];
        while p != (MAX - 1) {
            cap[p][c] -= VAL;
            cap[c][p] += VAL;
            c = p;
            p = parents[c];
        }
    }
    // XXX: Now just get the s-t cut
    fn dfs_st(
        adj: &[Vec<usize>],
        cap: &[Vec<i64>],
        s: usize,
        res: &mut Vec<usize>,
        vis: &mut [bool],
    ) {
        vis[s] = true;
        for &c in adj[s].iter() {
            if !vis[c] && cap[s][c] > 0 {
                dfs_st(adj, cap, c, res, vis);
            }
        }
        res.push(s);
    }
    let mut vis = vec![false; ss[0]];
    let mut res: Vec<_> = vec![];
    dfs_st(&adj, &cap, 0, &mut res, &mut vis);
    // XXX: Get the edges to cut
    let mut ress: Vec<(usize, usize)> = vec![];
    for (j, &r) in res.iter().enumerate() {
        let u = cap[r]
            .iter()
            .enumerate()
            .filter(|&(_, x)| *x == 0)
            .map(|(i, &_)| i)
            .filter(|&x| !res.iter().any(|&y| y == x));
        for k in u {
            ress.push((j + 1, k + 1));
        }
    }
    // XXX: Finally print the result
    println!("{}", ress.len());
    ress.iter().for_each(|&(i, j)| println!("{} {}", i, j));
}

#[derive(Debug, Clone, Default)]
struct SegNode<T>
where
    T: Default,
{
    _li: Us,
    _val: T,
    _ui: Us,
}

// XXX: f1 is the base case function. f2 is the merge case function
fn _build_seg_tree<F1, F2, T, U>(
    o: &[U],
    _stree: &mut [SegNode<T>],
    _vmap: &mut [usize],
    f1: &F1,
    f2: &F2,
    s: Us,
    e: Us,
    i: Us,
) where
    F1: for<'a> Fn(&'a T, &'a T) -> T,
    F2: for<'a> Fn(&'a U) -> T,
    T: Default + Copy,
    U: Default + Copy,
{
    // XXX: The base case
    if s == e - 1 {
        _stree[i] = SegNode {
            _li: s,
            _val: f2(&o[s]),
            _ui: e - 1,
        };
        _vmap[s] = i;
    } else {
        // XXX: The recursive case
        let mid = (e - s) / 2;
        // XXX: Do the left child
        _build_seg_tree(o, _stree, _vmap, f1, f2, s, s + mid, i * 2 + 1);
        // XXX: The the right child
        _build_seg_tree(o, _stree, _vmap, f1, f2, s + mid, e, i * 2 + 2);
        // XXX: Merge the outcome of the children
        _stree[i] = SegNode {
            _li: s,
            _val: f1(&_stree[i * 2 + 1]._val, &_stree[i * 2 + 2]._val),
            _ui: e - 1,
        }
    }
}

fn _seg_tree_query<F, T>(_stree: &[SegNode<T>], qil: Us, qiu: Us, f: &F, i: Us) -> T
where
    F: for<'a> Fn(&'a T, &'a T) -> T,
    T: Default + Copy,
{
    if qil == _stree[i]._li && qiu == _stree[i]._ui {
        // XXX: Return the value found -- base case
        _stree[i]._val
    } else if qil >= _stree[i * 2 + 1]._li && qiu <= _stree[i * 2 + 1]._ui {
        // XXX: Enter the left child
        _seg_tree_query(_stree, qil, qiu, f, i * 2 + 1)
    } else if qil >= _stree[i * 2 + 2]._li && qiu <= _stree[i * 2 + 2]._ui {
        // XXX: Enter the right child
        _seg_tree_query(_stree, qil, qiu, f, i * 2 + 2)
    } else {
        // XXX: Split the query into left and right child
        let ll = _seg_tree_query(_stree, qil, _stree[i * 2 + 1]._ui, f, i * 2 + 1);
        let rl = _seg_tree_query(_stree, _stree[i * 2 + 2]._li, qiu, f, i * 2 + 2);
        f(&ll, &rl)
    }
}

// XXX: This can be made much better using a simple while loop going
// upwards to parent.
fn _seg_tree_update<F, F2, T, U>(qi: Us, val: U, _stree: &mut [SegNode<T>], f: &F, f2: &F2, i: Us)
where
    F: for<'a> Fn(&'a T, &'a T) -> T,
    F2: for<'a> Fn(&'a U) -> T,
    T: Default,
{
    if qi == _stree[i]._li && qi == _stree[i]._ui {
        // XXX: Return the value found -- base case
        _stree[i]._val = f2(&val);
    } else if qi >= _stree[i * 2 + 1]._li && qi <= _stree[i * 2 + 1]._ui {
        // XXX: Enter the left child
        _seg_tree_update(qi, val, _stree, f, f2, i * 2 + 1);
        // XXX: Now update this value looking at the child values
        _stree[i]._val = f(&_stree[i * 2 + 1]._val, &_stree[i * 2 + 2]._val);
    } else if qi >= _stree[i * 2 + 2]._li && qi <= _stree[i * 2 + 2]._ui {
        // XXX: Enter the right child
        _seg_tree_update(qi, val, _stree, f, f2, i * 2 + 2);
        // XXX: Now update this value looking at the child values
        _stree[i]._val = f(&_stree[i * 2 + 1]._val, &_stree[i * 2 + 2]._val);
    } else {
        // XXX: This should never happen!
        panic!("Entered a wrong branch!");
    }
}

fn _seg_tree_q_by_val<T, F>(_stree: &[SegNode<T>], f: &F, val: T, i: Us) -> i64
where
    T: Default,
    F: Fn(&T, &T) -> bool,
{
    if f(&val, &_stree[i]._val) && _stree[i]._li == _stree[i]._ui {
        // XXX: Element found in the leaf node
        i as i64
    } else if f(&val, &_stree[i * 2 + 1]._val) {
        // XXX: Enter the left child first
        _seg_tree_q_by_val(_stree, f, val, i * 2 + 1)
    } else if f(&val, &_stree[i * 2 + 2]._val) {
        // XXX: Enter the right child
        _seg_tree_q_by_val(_stree, f, val, i * 2 + 2)
    } else {
        // XXX: The element required is just not there in the seg-tree
        -1
    }
}

fn _xor_query() {
    let ss = _read::<Us>();
    let _sg = _read::<Us>();
    let mut qs: Vec<(Us, Us)> = vec![(0, 0); ss[1]];
    for counter in 0..ss[1] {
        let m = _read::<Us>();
        qs[counter] = (m[0] - 1, m[1] - 1);
    }
    // XXX: Make a segment tree for _sg
    let h = (ss[0] as f64).log2().ceil() as Us;
    let mut i = 1;
    for j in 0..h {
        i += 2 << j;
    }
    let f = |x: &Us, y: &Us| x ^ y; // the xor function on usize
    let f2 = |x: &Us| *x;
    let mut _stree: Vec<SegNode<Us>> = vec![Default::default(); i];
    let mut _vmap: Vec<usize> = vec![0; ss[0]];
    _build_seg_tree(&_sg, &mut _stree, &mut _vmap, &f, &f2, 0, ss[0], 0);

    // XXX: Now process the queries.
    for &(qil, qiu) in qs.iter() {
        // println!("tutu: {} {}", qil, qiu);
        println!("{}", _seg_tree_query(&_stree, qil, qiu, &f, 0));
    }
}

fn _hotel_queries() {
    let ss = _read::<Us>();
    let hotels = _read::<Us>();
    let _rooms = _read::<Us>();
    let h = (hotels.len() as f64).log2().ceil() as Us;
    let i = (0..h).map(|x| 2 << x).sum::<Us>() + 1;
    let mut _stree: Vec<SegNode<Us>> = vec![Default::default(); i];
    let mut _vmap: Vec<Us> = vec![0; ss[0]];
    let f1 = |x: &Us, y: &Us| *max(x, y);
    let f2 = |x: &Us| *x;
    _build_seg_tree(&hotels, &mut _stree, &mut _vmap, &f1, &f2, 0, ss[0], 0);
    // println!("{:?}, {}", _stree, _stree.len());
    // XXX: Now query the node that has the first value that is required
    let f = |x: &Us, y: &Us| x <= y;
    let f2 = |x: &Us| *x;
    for &q in _rooms.iter() {
        let res = _seg_tree_q_by_val(&_stree, &f, q, 0);
        // println!("{res}");
        if res != -1 {
            // XXX: The update the value in the node with the new amount
            _seg_tree_update(
                _stree[res as usize]._li,
                _stree[res as usize]._val - q,
                &mut _stree,
                &f1,
                &f2,
                0,
            );
            print!("{} ", _stree[res as usize]._li + 1);
        } else {
            print!("{} ", res + 1);
        }
    }
    println!();
}

fn _max_array_sums() {
    let _ss = _read::<Us>();
    let _a = _read::<i64>();
    let mut qs: Vec<(Us, i64)> = vec![(0, 0); _ss[1]];
    for i in 0.._ss[1] {
        let mm = _read::<i64>();
        qs[i] = ((mm[0] - 1) as Us, mm[1]);
    }
    type Ps = i64;
    type Ss = i64;
    type Ms = i64;
    type Ts = i64;
    let h = (_ss[0] as f64).log2().ceil() as usize;
    let i = (0..h).map(|x| 2 << x).sum::<usize>() + 1;
    let mut _stree: Vec<SegNode<(Ps, Ss, Ms, Ts)>> = vec![Default::default(); i];
    let mut _vmap = vec![0usize; _ss[0]];
    // XXX: The merge function for max sums
    let f1 = |x: &(Ps, Ss, Ms, Ts), y: &(Ps, Ss, Ms, Ts)| {
        let ps = max(x.0, x.3 + y.0); // prefix sum
        let ss = max(y.1, y.3 + x.1); // suffix sum
        let ts = x.3 + y.3; // total sum
        let ms = max(x.2, y.2);
        let ms = max(ms, x.1 + y.0);
        (ps, ss, ms, ts)
    };
    let f2 = |x: &i64| (*x, *x, *x, *x);
    // XXX: Make the segment tree
    _build_seg_tree(&_a, &mut _stree, &mut _vmap, &f1, &f2, 0, _ss[0], 0);
    for &(k, v) in qs.iter() {
        _seg_tree_update(k, v, &mut _stree, &f1, &f2, 0);
        println!("{}", _stree[0]._val.2);
    }
}

fn _poly_queries() {
    let _ss = _read::<Us>();
    let _a = _read::<Us>();
    let mut _qs = vec![(0u8, 0usize, 0usize); _ss[1]];
    for _q in 0.._ss[1] {
        let _ss = _read::<usize>();
        _qs[_q] = (_ss[0] as u8, _ss[1], _ss[2]);
    }
    // XXX: Build the segment tree
    let h = (_ss[0] as f64).log2().ceil() as usize;
    let i = (0..h).map(|x| 2 << x).sum::<usize>() + 1;
    let mut _stree: Vec<SegNode<Us>> = vec![Default::default(); i];
    let mut _vmap = vec![0usize; _ss[0]];
    let f1 = |x: &Us, y: &Us| x + y;
    let f2 = |x: &Us| *x;
    _build_seg_tree(&_a, &mut _stree, &mut _vmap, &f1, &f2, 0, _ss[0], 0);
    // O(i) extra space used here
    let mut _vis: Vec<bool> = vec![false; i];
    // O(i) extra space used here -- max heap
    let mut _pq: BinaryHeap<Us> = BinaryHeap::with_capacity(i);
    // XXX: Now process the queries
    // XXX: O(|q|)
    for &(_q, _a, _b) in _qs.iter() {
        match _q {
            2 => println!("{}", _seg_tree_query(&_stree, _a - 1, _b - 1, &f1, 0)),
            1 => {
                let mut c = 1;
                // XXX: O(_ss[0])
                for i in _a - 1.._b {
                    let vv = _vmap[i]; // index check
                    _stree[vv]._val += c;
                    // XXX: Put the parent into the _pq
                    if !_vis[(vv - 1) / 2] {
                        _pq.push((vv - 1) / 2);
                        _vis[(vv - 1) / 2] = true;
                    }
                    c += 1;
                }
                // XXX: Now traverse upwards while _pq is not empty
                // XXX: O(log(_ss[0]))
                while !_pq.is_empty() {
                    let y = _pq.pop().unwrap();
                    let lc = &_stree[y * 2 + 1]._val;
                    let rc = &_stree[y * 2 + 2]._val;
                    _stree[y]._val = f1(lc, rc);
                    if y > 0 && !_vis[(y - 1) / 2] {
                        _vis[(y - 1) / 2] = true;
                        _pq.push((y - 1) / 2);
                    }
                }
            }
            _ => exit(1),
        }
    }
}

// XXX: Build a suffix automat
#[derive(Clone, Debug)]
struct SuffixNode {
    _next: HashMap<char, Us>,
    _len: Us,
    _slink: i64,
    _idx: Us,
    _fpos: Us,
}

impl Default for SuffixNode {
    fn default() -> Self {
        SuffixNode {
            _next: HashMap::default(),
            _len: 0,
            _slink: -1,
            _idx: 0,
            _fpos: 0,
        }
    }
}

fn _build_suffix_automata(_s: &str, _v: &mut Vec<SuffixNode>) -> Us {
    let mut _last = 0i64; // The first node
    let mut _ss = 1usize; // The total number of nodes in the
                          // suffix automata
    for _c in _s.chars() {
        // println!("Putting char: {_c}");
        let _curr = _ss; // index of the curr node

        // XXX: First make a new node
        let mut nnode = SuffixNode::default();
        nnode._len += _v[_last as Us]._len + 1;
        nnode._idx = _curr;
        nnode._fpos = nnode._len - 1;
        _v.push(nnode);

        _ss += 1; // total number of nodes increased by 1

        // XXX: Now traverse back to make connections
        let mut _p = _last;
        while _p != -1 && !_v[_p as Us]._next.contains_key(&_c) {
            let n = &mut _v[_p as Us];
            n._next.insert(_c, _curr);
            _p = n._slink;
        }
        if _p == -1 {
            // XXX: Just add suffix link for current node
            _v[_curr]._slink = 0;
        } else {
            // XXX: This means there is another node with edge with
            // label _c
            let _q = _v[_p as Us]._next[&_c];
            // println!("_q is: {_q}");
            if _v[_q]._len == _v[_p as Us]._len + 1 {
                _v[_curr]._slink = _q as i64;
            } else {
                // XXX: Make a clone of q
                let mut _clone = _v[_q].clone();
                // println!("cloned node: {:?}", _clone);
                let _clone_idx = _ss;
                // println!("cloned index: {:?}", _clone_idx);
                _ss += 1;
                _clone._len = _v[_p as Us]._len + 1;
                _clone._fpos = _clone._len - 1;
                _clone._idx = _clone_idx;
                // println!("cloned node after update: {:?}", _clone);
                _v.push(_clone);
                while _p != -1 && _v[_p as Us]._next[&_c] == _q {
                    *_v[_p as Us]._next.get_mut(&_c).unwrap() = _clone_idx;
                    _p = _v[_p as Us]._slink;
                }
                // XXX: Finally update the suffix links for _q and _clone
                _v[_q]._slink = _clone_idx as i64;
                _v[_curr]._slink = _clone_idx as i64;
            }
        }
        _last = _curr as i64;
    }
    _last as Us
}

fn _paths_to_term_states(ps: &mut [Us], ts: &[bool], am: &[SuffixNode], s: Us) {
    if ts[s] {
        ps[s] += 1;
    }
    for (&_, &v) in am[s]._next.iter() {
        if ps[v] == 0 {
            _paths_to_term_states(ps, ts, am, v);
        }
        ps[s] += ps[v];
    }
}

fn _terminal_nodes(vec: &[SuffixNode], tnodes: &mut [bool], s: Us) {
    let mut p = s;
    while p != 0 {
        tnodes[p] = true;
        p = vec[p]._slink as Us;
    }
}

fn _string_matching() {
    let _ss = _read::<String>();
    let _p = _read::<String>();

    // XXX: Make the initial node i
    let _i = SuffixNode::default();
    // XXX: The vector where the suffix automata will be kept
    let mut _vec: Vec<SuffixNode> = Vec::with_capacity(2 * _ss[0].len());
    // XXX: Push the initial node in position 0
    _vec.push(_i);
    // XXX: Now build the suffix automata
    let last = _build_suffix_automata(&_ss[0], &mut _vec);
    // XXX: Get the terminal nodes in the suffix automata
    let mut term_nodes = vec![false; _vec.len()];
    _terminal_nodes(&_vec, &mut term_nodes, last);

    // XXX: Get the paths to terminal node(s) for each state in the
    // automata
    let mut _path_to_terms = vec![0; _vec.len()];
    _paths_to_term_states(&mut _path_to_terms, &term_nodes, &_vec, 0);
    // XXX: Now get the matching result
    let mut vv = 0usize;
    for c in _p[0].chars() {
        vv = *_vec[vv]._next.get(&c).unwrap();
    }
    println!("{}", _path_to_terms[vv]);
}

fn _find_patterns() {
    let ss = &_read::<String>()[0];
    let n = _read::<usize>()[0];
    // XXX: Build the suffix tree
    let mut _vec: Vec<SuffixNode> = Vec::with_capacity(2 * ss.len());
    // XXX: Push the first node
    _vec.push(SuffixNode::default());
    // XXX: Build the suffix automata
    _build_suffix_automata(ss, &mut _vec);
    println!();
    for _ in 0..n {
        let mut vv = 0usize;
        let mut done = true;
        let o = &_read::<String>()[0];
        for c in o.chars() {
            match _vec[vv]._next.get(&c) {
                Some(&x) => vv = x,
                None => {
                    done = false;
                    break;
                }
            }
        }
        if !done {
            println!("NO");
        } else {
            println!("YES");
        }
    }
}

fn _counting_patterns() {
    let ss = &_read::<String>()[0];
    let n = _read::<usize>()[0];
    // XXX: Build the suffix tree
    let mut _vec: Vec<SuffixNode> = Vec::with_capacity(2 * ss.len());
    // XXX: Push the first node
    _vec.push(SuffixNode::default());
    // XXX: Build the suffix automata
    let last = _build_suffix_automata(ss, &mut _vec);
    let mut term_nodes = vec![false; _vec.len()];
    _terminal_nodes(&_vec, &mut term_nodes, last);

    // XXX: Get the paths to terminal node(s) for each state in the
    // automata
    let mut _path_to_terms = vec![0; _vec.len()];
    _paths_to_term_states(&mut _path_to_terms, &term_nodes, &_vec, 0);
    // XXX: Get the number of paths to the terminal nodes
    println!();
    for _ in 0..n {
        let mut vv = 0usize;
        let mut done = true;
        let o = &_read::<String>()[0];
        for c in o.chars() {
            match _vec[vv]._next.get(&c) {
                Some(&x) => vv = x,
                None => {
                    done = false;
                    break;
                }
            }
        }
        if !done {
            println!("0");
        } else {
            println!("{}", _path_to_terms[vv]);
        }
    }
}

fn _d_paths(_vec: &[SuffixNode], _p: &mut [Us], s: Us) {
    for (&_, &v) in _vec[s]._next.iter() {
        if _p[v] == 0 {
            _d_paths(_vec, _p, v);
        }
        _p[s] += _p[v];
    }
    if s != 0 {
        _p[s] += 1;
    }
}

fn _distinct_substrings() {
    let ss = &_read::<String>()[0];
    let mut _vec: Vec<SuffixNode> = Vec::with_capacity(ss.len() * 2);
    _vec.push(SuffixNode::default());
    _build_suffix_automata(ss, &mut _vec);
    // XXX: Get all edges to the last node
    let mut _paths = vec![0; _vec.len()];
    _d_paths(&_vec, &mut _paths, 0);
    println!("{}", _paths[0]);
}

fn _longest_path_to_term(_v: &[SuffixNode], _paths: &mut [Us], s: Us) {
    let mut tp = _paths[s];
    for (&_, &v) in _v[s]._next.iter() {
        if _paths[v] == 0 {
            _longest_path_to_term(_v, _paths, v);
        }
        tp = max(1 + _paths[v], _paths[s]);
    }
    _paths[s] = tp;
}

fn _pattern_position() {
    let ss = &_read::<String>()[0];
    let n = _read::<Us>()[0];
    let mut _vec: Vec<SuffixNode> = Vec::with_capacity(ss.len() * 2);
    _vec.push(SuffixNode::default());
    _build_suffix_automata(ss, &mut _vec);
    let mut _paths = vec![0; _vec.len()];
    _longest_path_to_term(&_vec, &mut _paths, 0);
    println!();
    for _ in 0..n {
        let mut counter = 0usize;
        let mut vv = 0usize;
        let o = &_read::<String>()[0]; //The pattern
        let mut done = true;
        for c in o.chars() {
            match _vec[vv]._next.get(&c) {
                Some(&x) => {
                    vv = x;
                    counter += 1;
                }
                None => {
                    done = false;
                    break;
                }
            }
        }
        if !done {
            println!("-1");
        } else {
            println!("{}", ss.len() - _paths[vv] - counter + 1);
        }
    }
}

fn _palindrome_query() {
    let _ns = _read::<Us>();
    let _ss = &mut _read::<String>()[0];
    // XXX: Read the queries
    for _ in 0.._ns[1] {
        let q: Vec<String> = _read::<String>();
        let qq: Us = q[0].parse().unwrap();
        match qq {
            2 => {
                let i: Us = q[1].parse().unwrap();
                let j: Us = q[2].parse().unwrap();
                _is_palindrome(&_ss[i - 1..j]);
            }
            1 => {
                let i = q[1].parse::<Us>().unwrap();
                let v: char = q[2].parse().unwrap();
                unsafe {
                    let vv = _ss.as_mut_vec();
                    vv[i - 1] = v as u8;
                }
            }
            _ => (),
        }
    }
    fn _is_palindrome(s: &str) {
        let mut done = true;
        let mut cc = 0;
        for (c1, c2) in zip(s.chars(), s.chars().rev()) {
            if c1 != c2 {
                done = false;
                break;
            }
            if cc == s.len() / 2 {
                break;
            }
            cc += 1;
        }
        if done {
            println!("YES");
        } else {
            println!("NO");
        }
    }
}

fn _substr_dist() {
    let _ss = &_read::<String>()[0];
    let mut _vec: Vec<SuffixNode> = Vec::with_capacity(_ss.len() * 2);
    _vec.push(SuffixNode::default());
    _build_suffix_automata(&_ss, &mut _vec);
    let mut freq = vec![0usize; _ss.len() + 1];
    // XXX: Now just get the distribution of each substring
    fn _freq_dis(v: &[SuffixNode], _f: &mut [Us], c: Us, i: Us) {
        if c > 0 {
            _f[c] += 1;
        }
        for (&_, &vv) in v[i]._next.iter() {
            _freq_dis(v, _f, c + 1, vv);
        }
    }
    _freq_dis(&_vec, &mut freq, 0, 0);
    freq[1..].iter().for_each(|&x| print!("{} ", x));
    println!();
}

fn _hamming_distance() {
    let ss = _read::<Us>();
    let mut _vv: Vec<Us> = vec![0; ss[0]];
    for i in 0..ss[0] {
        let _m = &_read::<String>()[0];
        _vv[i] = Us::from_str_radix(_m, 2).unwrap();
    }
    // XXX: Now do hamming distance
    let mut res: Us = MAX;
    for (_c, &_i) in _vv.iter().enumerate() {
        for &_j in _vv[_c + 1..].iter() {
            let _y = _i ^ _j;
            res = min(res, _y.count_ones() as usize);
        }
    }
    println!("{res}");
}

// XXX: This is the Hungarian Algorithm.
fn _task_assignment() {
    let n = _read::<Us>()[0];
    let mut t: Vec<Vec<Us>> = vec![vec![0; n]; n];
    for i in 0..n {
        let ss = _read::<Us>();
        t[i] = ss; //this is a move operation, not copy
    }
    let _orig = t.clone(); // just kept for later

    // XXX: Now sub the min for each row first and then column.
    // XXX: Row subtraction
    for i in 0..n {
        let &m = t[i].iter().min().unwrap();
        t[i] = t[i].iter().map(|x| x - m).collect::<Vec<_>>();
    }

    // XXX: Column subtraction -- this code sucks!
    for j in 0..n {
        let mut m = MAX;
        for i in 0..n {
            m = min(t[i][j], m);
        }
        // XXX: Now subtract the minimum
        for i in 0..n {
            t[i][j] -= m;
        }
    }
    let mut cols = vec![-1i64; n];
    for i in 0..n {
        _g_match(&t, &mut cols, 0, i, 0);
    }
    while cols.iter().any(|&x| x == -1) {
        // XXX: Then we have to do more work

        // XXX: The unassigned rows
        let _urows = (0..n)
            .filter(|&x| !cols.iter().any(|&y| x == y as Us))
            .collect::<Vec<_>>();
        // XXX: The minimum delta
        let mut _d = MAX;
        // XXX: The rows that have been assigned to these columns
        let mut _ras: Vec<Us> = Vec::with_capacity(n);
        for &j in _urows.iter() {
            _ras.clear();
            // XXX: Get the zeros columns
            let _zc = t[j]
                .iter()
                .enumerate()
                .filter(|(_, &x)| x == 0)
                .map(|(t, &_)| t)
                .collect::<Vec<_>>();

            for &k in _zc.iter() {
                let _ra = cols[k];
                assert!(_ra != -1);
                _ras.push(_ra as Us);
                // XXX: The the minimum from amongst all rows that are
                // not in _zc.
                let &_d1 = t[j].iter().filter(|&&x| x != 0).min().unwrap();
                let &_d2 = t[_ra as Us].iter().filter(|&&x| x != 0).min().unwrap();
                _d = min(_d1, _d2);
            }
            assert!(_d != 0);
            _ras.push(j); // just making it easy to iterate

            // XXX: Now do the pivot for simplex
            for &h in &_ras {
                for l in 0..n {
                    if t[h][l] > 0 {
                        t[h][l] -= _d;
                    }
                }
            }
            for &h in &cols {
                if h != -1 {
                    for &l in &_zc {
                        if t[h as Us][l] > 0 {
                            t[h as Us][l] += _d;
                        }
                    }
                }
            }
        }
        // XXX: Now carry out the greedy match again
        cols.fill(-1);
        for i in 0..n {
            _g_match(&t, &mut cols, 0, i, 0);
        }
    }
    // XXX: We are done
    let mut total = 0;
    for i in 0..n {
        total += _orig[cols[i] as Us][i];
    }
    println!("{total}");
    for (i, &x) in cols.iter().enumerate().take(n) {
        println!("{} {}", x + 1, i + 1);
    }
}

fn _new_road_qs() {
    let ss = _read::<Us>();
    let mut map = (1..ss[0] + 1).map(|x| (x, 0)).collect::<HashMap<Us, Us>>();
    for i in 0..ss[1] {
        let mm = _read::<Us>();
        if map[&mm[0]] == 0 {
            *map.get_mut(&mm[0]).unwrap() = i + 1;
        }
        if map[&mm[1]] == 0 {
            *map.get_mut(&mm[1]).unwrap() = i + 1;
        }
    }
    println!();
    // XXX: Now the queries
    for _ in 0..ss[2] {
        let mm = _read::<Us>();
        if map[&mm[0]] == 0 || map[&mm[1]] == 0 {
            println!("-1");
        } else {
            println!("{}", max(map[&mm[0]], map[&mm[1]]));
        }
    }
}

fn _exponent() {
    let n = _read::<Us>()[0];
    let _y = 10usize.pow(9) + 7;
    for _ in 0..n {
        let ss = _read::<Us>();
        println!("{}", (0..ss[1]).fold(1, |acc, _| acc * ss[0]) % _y);
    }
}

fn _fib() {
    let _ss = _read::<Us>()[0];
    if _ss <= 3 {
        println!("2");
    } else {
        let _y = 10usize.pow(9) + 7;
        let mut _oo = 1usize; // for fib(2)
        let mut _o = 2usize; // for fib(3)
        for _ in 3.._ss {
            let u = _o + _oo;
            _oo = _o;
            _o = u;
        }
        println!("{}", _o % _y);
    }
}

fn _fib_golden_ratio() {
    let n = _read::<Us>()[0];
    let phi = (1f64 + 5f64.sqrt()) / 2f64;
    let si = (1f64 - 5f64.sqrt()) / 2f64;
    let phi_p = phi.powi(n as i32);
    let si_p = si.powi(n as i32);
    let res = ((phi_p - si_p) / (phi - si)).ceil() as Us;
    let _y = 10usize.pow(9) + 7;
    println!("{}", res % _y);
}

fn _transport_assignment() {
    let ss = _read::<Us>();
    let mut costs = vec![vec![MAX; ss[0]]; ss[0]];
    let s = ss[0];
    let t = s + 1;
    let mut cap = vec![vec![-1i64; ss[0] + 2]; ss[0] + 2];
    for _ in 0..ss[1] {
        // XXX: Update the costs and capacity matrix
        let m = _read::<Us>();
        let ss = m[0] - 1;
        let tt = m[1] - 1;
        let ca = m[2];
        let co = m[3];
        costs[ss][tt] = co;
        costs[tt][ss] = co;
        cap[ss][tt] = ca as i64;
        cap[tt][ss] = 0;
    }
    let _orig = costs.clone(); // needed for final cost calculation
    let _ocap = cap.clone(); // needed for final calculation

    // XXX: Add the capacity for source and target
    cap[s][0] = ss[2] as i64;
    cap[0][s] = 0;
    cap[ss[0] - 1][t] = ss[2] as i64;
    cap[t][ss[0] - 1] = 0;
    // println!("costs: {:?}", costs);
    // println!("capacity: {:?}", cap);

    // XXX: Now subtract the min cost for each row
    for i in 0..ss[0] {
        let mm = *costs[i].iter().min().unwrap();
        costs[i] = costs[i].iter().map(|&x| x - mm).collect::<Vec<_>>();
    }
    // XXX: Subtract costs for min column
    for j in 0..ss[0] {
        let mut mm = MAX;
        for i in 0..ss[0] {
            mm = min(mm, costs[i][j]);
        }
        for i in 0..ss[0] {
            costs[i][j] -= mm;
        }
    }
    // println!("costs: {:?}", costs);
    // println!("capacity: {:?}", cap);

    // XXX: Make the adj list for the 0 cost edges
    fn add_to_adj(adj: &mut [Vec<Us>], s: Us, t: Us, ss: &[Us], costs: &[Vec<Us>]) {
        for i in 0..ss[0] {
            adj[i] = costs[i]
                .iter()
                .enumerate()
                .filter(|(_, &x)| x == 0)
                .map(|(t, &_)| t)
                .collect::<Vec<_>>();
        }
        // XXX: Add the source and target nodes too
        adj[s].push(0);
        adj[0].push(s);
        adj[ss[0] - 1].push(t);
        adj[t].push(ss[0] - 1);
    }
    // println!("{:?}", adj);

    let mut parents = vec![MAX; ss[0] + 2];
    // XXX: Do max-flow on the adj graph

    let mut adj: Vec<Vec<Us>> = vec![vec![]; ss[0] + 2];
    let mut flow: i64;
    let mut tflow: i64 = 0;
    loop {
        adj.iter_mut().for_each(|x| x.clear());
        add_to_adj(&mut adj, s, t, &ss, &costs);
        flow = _bfs_max_flow(s, t, &mut parents, &adj, &cap);
        tflow += flow;
        if flow == 0 {
            break;
        }
        // XXX: Update the capacity matrix
        let mut c: usize = t;
        let mut p: usize = parents[c];
        while p != (MAX - 1) {
            cap[p][c] -= flow;
            cap[c][p] += flow;
            c = p;
            p = parents[c];
        }
        // XXX: Add other edges not present in the graph
        let mm = *costs[ss[0] - 1].iter().filter(|&&x| x > 0).min().unwrap();
        costs[ss[0] - 1] = costs[ss[0] - 1]
            .iter()
            .map(|&x| if x == 0 { x } else { x - mm })
            .collect::<Vec<_>>();
        // XXX: Do the same for the column [t]
        for j in 0..ss[0] {
            if costs[j][ss[0] - 1] > 0 {
                costs[j][ss[0] - 1] -= mm;
            }
        }
    }
    if tflow != ss[2] as i64 {
        println!("-1");
    } else {
        let mut total_cost = 0i64;
        for i in 0..ss[0] {
            for j in (i + 1)..ss[0] {
                total_cost += (_ocap[i][j] - cap[i][j]) * (_orig[i][j] as i64);
            }
        }
        println!("{:?}", total_cost);
    }
}

fn _heapify<T, F>(a: &mut [T], _n: Us, l: Us, f: &F)
where
    T: std::fmt::Debug,
    F: for<'a> Fn(&'a T, &'a T) -> bool,
{
    // XXX: First get the element that satisfies the function f
    let mut largest = l;
    // XXX: You need the extra < n, because we are doing place swaps
    largest = if (l * 2 + 1) < _n && f(&a[l * 2 + 1], &a[largest]) {
        l * 2 + 1
    } else {
        largest
    };
    largest = if (l * 2 + 2) < _n && f(&a[l * 2 + 2], &a[largest]) {
        l * 2 + 2
    } else {
        largest
    };
    // XXX: Got the largest value amongst the subtree
    if largest != l {
        let a_ptr = a.as_mut_ptr();
        unsafe {
            let (x, y) = (a_ptr.add(l), a_ptr.add(largest));
            swap(x, y);
        }
    }

    // XXX: Now heapify the subtree from largest
    if largest * 2 + 1 < _n && largest != l {
        _heapify(a, _n, largest, f);
    }
}

fn _heap_sort() {
    let mut _ss = _read::<i64>();
    let _n: i64 = _ss.len() as i64 - 1;
    // XXX: Sort this given vector (ss)
    let f = |x: &i64, y: &i64| x >= y;
    let mut j: i64 = if _n % 2 == 0 {
        (_n - 2) / 2
    } else {
        (_n - 1) / 2
    };
    while j >= 0 {
        _heapify(&mut _ss, (_n + 1) as Us, j as Us, &f);
        j -= 1;
    }
    // XXX: Now sort them
    for i in (0.._n + 1).rev() {
        let temp = _ss[i as Us];
        _ss[i as Us] = _ss[0];
        _ss[0] = temp;
        _heapify(&mut _ss, i as Us, 0, &f);
    }
    println!("{:?}", _ss);
}

fn _quick_sort() {
    let mut _ss = _read::<Us>();
    fn _partition<T>(a: &mut [T], _e: Us, _i: Us)
    where
        T: PartialOrd + std::fmt::Debug,
    {
        if _e - _i <= 0 {
            return;
        }
        let mut _pp = _e - 1;
        let mut j = _i;
        while j < _e {
            if j <= _pp && a[j] >= a[_pp] {
                // XXX: Then swap
                let a_ptr = a.as_mut_ptr();
                unsafe {
                    let (x, y) = ((a_ptr.add(j)), (a_ptr.add(_pp)));
                    swap(x, y);
                }
            } else if j > _pp && a[j] <= a[_pp] {
                let a_ptr = a.as_mut_ptr();
                unsafe {
                    let (x, y) = ((a_ptr.add(j)), (a_ptr.add(_pp)));
                    swap(x, y);
                }
                let op = _pp;
                _pp = j;
                j = op;
            }
            j += 1;
        }
        _partition(a, _pp, _i);
        _partition(a, _e, _pp + 1);
    }
    let n = _ss.len();
    _partition(&mut _ss, n, 0);
    println!("{:?}", _ss);
}

fn main() {
    // XXX: Beginner problems
    // _weird_algo();
    // _two_sets();

    // XXX: Graph algorithms
    // _building_teams(); //bi-partite graph
    // _course_schedule(); //topological sort
    // _longest_flight_route(); //longest path with dynamic programming
    // _planets_qs1(); //just reachability
    // _game_routes(); //just dp for number of paths to destination
    // _road_reparation(); // kruskal' algo for minimum spanning tree
    // _flight_routes_check(); // strongly connected components, Kosaraju' alog
    // _teleporters_path(); //eulerian path
    // _hamiltonian_flights(); //hamiltonian path, NP-hard, without memoization
    // _giant_pizza(); //2-sat problem, implication graph with scc
    // _school_dance(); //maximum matching for bi-partite graph -- greedy algo
    // _police_chase(); // max-flow min s-t cut, ford-fulkerson algorithm

    // XXX: Range queries -- segment tree examples for array range queries.
    // _xor_query(); // xor segment tree
    // _hotel_queries(); // max segment tree with updates
    // _max_array_sums(); // max prefix/suffix/total sums

    // This is a runtime efficient update of polynomial queries.
    // However, it does use extra memory for a vector<bool> and a max
    // heap.
    // _poly_queries();

    // XXX: String processing
    // _string_matching(); // Number of repetitions of a pattern in a string
    // _find_patterns(); // Just find if the patterns exist in the suffix automata
    // _counting_patterns(); // Find the patterns, and count how many times
    // _distinct_substrings(); // The total number of distinct substrings
    // _pattern_position(); //Get the first "position" of each pattern
    // _palindrome_query(); //Check if a substring is a palindrome
    // _substr_dist(); //distribution of substrings in a string

    // XXX: Advanced algorithms
    // _hamming_distance(); // converting binary to decimal, popcount assembly inst
    // _new_road_qs(); // disjoint union of sets using hashmap
    // _task_assignment(); //This is the NxN task assignment problem -- hungarian algo
    // _transport_assignment(); // This is a optimal transportation problem

    // XXX: Mathematics
    // _exponent();
    // _fib(); // with dynamic programming
    // _fib_golden_ratio(); //Fibonacci formula using golden ratio

    // XXX: Sorting algorithms in rust
    _heap_sort();
    // _quick_sort();
}
