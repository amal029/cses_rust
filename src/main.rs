// XXX: This file does many algorithms from CSES website.
// Website: https://cses.fi/problemset/list/

use std::cmp::Reverse;
use std::collections::BinaryHeap;
use std::process::exit;

// XXX: Just made a generic reader for all problems
fn _read<T>() -> Vec<T>
where
    T: std::str::FromStr,
{
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line
        .trim()
        .split(' ')
        .flat_map(str::parse::<T>)
        .collect::<Vec<_>>();
    return nums;
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
    vs.reserve(n as usize);
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
        } else if cc.iter().any(|&x| x == 2) {
            // println!("giving color 1");
            crs[i] = 1;
        } else if cc.iter().all(|&x| x == 0) {
            // println!("giving color 1");
            crs[i] = 1;
        }
        // XXX: Now do this for its neighbors.
        let mut dd = true;
        for c1 in vs[i].iter() {
            if crs[*c1] == 0 {
                if !dfs_color(*c1, vs, crs) {
                    dd = false;
                    break;
                }
            }
        }
        return dd;
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
    let mut order = vec![];
    order.reserve(n);
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
        return dd;
    }
    for c in 0..n {
        if !vis[c] {
            if !top_sort(&adj, &mut order, &mut vis, c) {
                println!("IMPOSSIBLE");
                return;
            }
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
    for counter in 0..q {
        line.clear();
        std::io::stdin().read_line(&mut line).expect("input");
        let nums = line
            .trim()
            .split(' ')
            .flat_map(str::parse::<usize>)
            .collect::<Vec<_>>();
        qs[counter] = (nums[0] - 1, nums[1]);
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
        let (c, s, d) = match pq.pop().unwrap() {
            Reverse(x) => x,
        };
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
    if !vis.iter().all(|&x| x == true) {
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
    // println!("pathss len: {}", paths.len());
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
        println!("{} {}", sccs[0][0]+1, sccs[1][0]+1);
    }
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
    _flight_routes_check(); // strongly connected components, Kosaraju' alog
}
