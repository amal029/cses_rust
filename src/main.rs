// XXX: This file does many algorithms from CSES website.
// Website: https://cses.fi/problemset/list/

use std::process::exit;

fn _weird_algo() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line.trim().split(' ')
	.flat_map(str::parse::<i32>).collect::<Vec<_>>();
    let mut n : i32 = nums[0];
    print!("{} ", n);
    while n != 1{
	if n % 2 == 0{
	    n /= 2;
	} else{
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
    let nums = line.trim().split(' ')
	.flat_map(str::parse::<i32>).collect::<Vec<_>>();
    let mut n : i32 = nums[0];
    let on = n as usize;
    // XXX: If n*(n+1)/2 % 2 == 0 --> it is divisible.
    let sum = n*(n+1)/2;
    if sum % 2 == 0{
	println!("YES");
	let y = sum/2;
	let mut v1 = vec![];
	let mut ss : i32 = v1.iter().sum();
	while ss != y {
	    if ss + n <= y{
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
	    None => exit(1)
		
	};
	for i in *start+1..(n+1){
	    print!("{} ", i);
	}
	println!();
    }else {
	println!("NO");
    }
}

// XXX: This is checking if the graph is bi-partite.
fn _building_teams() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line.trim().split(' ')
	.flat_map(str::parse::<usize>).collect::<Vec<_>>();
    let n: usize = nums[0];
    let m: usize = nums[1];
    let mut vs : Vec<Vec<usize>> = vec![vec![]; n];
    vs.reserve(n as usize);
    let mut counter = 0;
    while counter < m{
	line.clear();
	std::io::stdin().read_line(&mut line).expect("input");
	let nums = line.trim().split(' ')
	    .flat_map(str::parse::<usize>).collect::<Vec<_>>();

	let mut s : usize = nums[0];
	s -= 1;
	let mut e : usize = nums[1];
	e -= 1;
	vs[s].push(e);
	vs[e].push(s);
	counter += 1;
    }
    // XXX: Now color the vertices with 1 or 2
    let mut colors : Vec<usize> = vec![0; n];
    fn dfs_color(i:usize, vs:&Vec<Vec<usize>>, crs: &mut Vec<usize>) -> bool {
	// println!("Doing : {i}");
	// println!("{:?}", crs);
	let mut cc = vec![];
	// XXX: All loop borrow via reference
	for c in vs[i].iter() {
	    // println!("{:?} for {:?}", crs[*c], *c);
	    cc.push(crs[*c]);
	}
	if !cc.iter().all(|&x| x == 0 || x == 1 || x == 2){
	    return false;
	} else if cc.iter().any(|&x| x == 1){
	    // println!("giving color 2");
	    crs[i] = 2;
	} else if cc.iter().any(|&x| x == 2){
	    // println!("giving color 1");
	    crs[i] = 1;
	} else if cc.iter().all(|&x| x == 0){
	    // println!("giving color 1");
	    crs[i] = 1;
	}
	// XXX: Now do this for its neighbors.
	let mut dd = true;
	for c1 in vs[i].iter(){
	    if crs[*c1] == 0{
		if !dfs_color(*c1, vs, crs){
		    dd = false;
		    break;
		}
	    }
	}
	return dd;
    }
    // XXX: Do this for all nodes
    for nodes in 0..n {
	if !dfs_color(nodes, &vs, &mut colors){
	    println!("IMPOSSIBLE");
	    break;
	}
    }
    for c in colors.iter(){
	print!("{c} ");
    }
    println!();
}

fn _course_schedule() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line.trim().split(' ')
	.flat_map(str::parse::<usize>).collect::<Vec<_>>();
    let n  = nums[0];
    let m = nums[1];
    let mut adj : Vec<Vec<usize>> = vec![vec![]; n];
    let mut counter = 0;
    while counter < m {
	line.clear();
	std::io::stdin().read_line(&mut line).expect("input");
	let nums = line.trim().split(' ')
	    .flat_map(str::parse::<usize>).collect::<Vec<_>>();
	let mut s :usize = nums[0];
	let mut e :usize = nums[1];
	s -= 1;
	e -= 1;
	// XXX: This is a directed graph
	adj[s].push(e);
	counter += 1;
    }
    // XXX: Do a topological sort of the DAG, we need to make check that
    // the graph is acyclic while traversing it.
    let mut order = vec![];
    order.reserve(n);
    let mut vis = vec![false; n];
    fn top_sort(adj: &Vec<Vec<usize>>, order: &mut Vec<usize>,
		vis : &mut Vec<bool>, i: usize) -> bool {
	vis[i] = true;
	// XXX: Now do a post-order traversal of the graph
	let mut dd = true;
	for c in adj[i].iter() {
	    if !vis[*c] {
		if !top_sort(adj, order, vis, *c){
		    dd = false;
		    break;
		}
	    } else{
		// XXX: Check that if it is already visited then it
		// should be in the order vector.
		match order.iter().find(|&x| x == c){
		    Some (_) => (),
		    None => return false
		}
	    }
	}
	if dd{
	    order.push(i);
	}
	return dd;
    }
    for c in 0..n {
	if !vis[c]{
	    if !top_sort(&adj, &mut order, &mut vis, c){
		println!("IMPOSSIBLE");
		return;
	    }
	}
    }
    
    for c in order.iter().rev(){
	print!("{} ", c+1);
    }
    println!();
    
}

fn _longest_flight_route() {
    let mut line = String::new();
    std::io::stdin().read_line(&mut line).expect("input");
    let nums = line.trim().split(' ')
	.flat_map(str::parse::<usize>).collect::<Vec<_>>();
    let n : usize = nums[0];
    let m : usize = nums[1];
    let mut counter = 0;
    let mut adj : Vec<Vec<usize>> = vec![vec![]; n];
    while counter < m {
	line.clear();
	std::io::stdin().read_line(&mut line).expect("input");
	let nums = line.trim().split(' ')
	    .flat_map(str::parse::<usize>).collect::<Vec<_>>();
	let mut s = nums[0];
	s -= 1;
	let mut e  = nums[1];
	e -= 1;
	// XXX: A DAG
	adj[s].push(e);
	counter += 1;
    }
    // XXX: Now just do DP for finding the longest path via DFS
}

fn main() {
    // XXX: Beginner problems
    // _weird_algo();
    // _two_sets();


    // XXX: Graph algorithms
    // _building_teams();
    // _course_schedule();
    _longest_flight_route();
}
