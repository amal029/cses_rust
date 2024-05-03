// XXX: This file does many algorithms from CSES website.
// Website: https://cses.fi/problemset/list/

use std::cmp::Reverse;
use std::collections::{BinaryHeap, VecDeque};
use std::process::exit;
use std::usize::MAX;

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

fn _greedy_match(m: &[Vec<usize>], cols: &mut [i64], tom: usize, row: usize, col: usize) -> bool {
    // XXX: Move through each row and get the first to_match.
    let cc = m[row]
        .iter()
        .enumerate()
        .filter(|&(t, x)| *x == tom && t >= col)
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
        for _j in cc.iter() {
            // XXX: Try moving one by one if possible
            done = _greedy_match(m, cols, tom, cols[*_j] as usize, *_j + 1);
            if done {
                // XXX: Add yourself to the cols
                cols[*_j] = row as i64;
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
        _greedy_match(&adj, &mut cols, 1, i, 0);
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

fn _bfs_max_flow(s: usize, t: usize, ps: &mut [usize], adj: &[Vec<usize>], cap: &[Vec<i8>]) -> i8 {
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
    flow as i8
}

fn _police_chase() {
    const VAL: i8 = 1i8;
    let ss = _read::<usize>();
    let mut adj: Vec<Vec<usize>> = vec![vec![]; ss[0]];
    let mut cap: Vec<Vec<i8>> = vec![vec![-1; ss[0]]; ss[0]];
    for _ in 0..ss[1] {
        let s = _read::<usize>();
        adj[s[0] - 1].push(s[1] - 1);
        adj[s[1] - 1].push(s[0] - 1);
        cap[s[0] - 1][s[1] - 1] = VAL;
        cap[s[1] - 1][s[0] - 1] = 0;
    }

    // XXX: Now just do ford-fulkerson algorithm
    let mut flow: i8;
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
        cap: &[Vec<i8>],
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
            .map(|(i, &_)| i);
        let _y = u.filter(|&x| !match res.iter().find(|&&y| y == x) {
            Some(_) => true,
            None => false,
        });
        for k in _y {
            ress.push((j + 1, k + 1));
        }
    }
    // XXX: Finally print the result
    println!("{}", ress.len());
    ress.iter().for_each(|&(i, j)| println!("{} {}", i, j));
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

    // XXX: String algorithms
}
