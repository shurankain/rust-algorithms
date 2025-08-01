# Rust Algorithms

A modular Rust workspace containing implementations of **classic algorithms** across multiple categories.  
The goal is to **learn, understand, and document algorithmic approaches** while deepening knowledge of the Rust programming language.  

---

## Categories

Currently implemented (or in progress):

- Backtracking  
- Cryptography  
- Divide and Conquer  
- Dynamic Programming  
- Geometry  
- Graphs  
- Greedy  
- Hashing  
- Machine Learning (WIP)  
- Searching  
- Sorting  
- Strings  
- Shared Utilities (WIP)  

---

## Structure

Uses Cargo workspaces to isolate algorithms by topic:

```bash
rust-algorithms/
|- Cargo.toml              # root workspace
|- README.md
|- sorting/                # sorting algorithms
|  |- src/
|     |- lib.rs
|- searching/              # search algorithms
|- graphs/                 # graph algorithms
|- dynamic_programming/
|- greedy/
|- divide_and_conquer/
|- backtracking/
|- hashing/
|- strings/
|- geometry/
|- crypto/
|- ml_basics/              # WIP
|- shared_utils/           # commons structures (Graph, Matrix e.g.) (WIP)
|- playground/             # binary for examples execution
````

---

## Usage

Run or test a specific algorithm using the `playground` binary:

```bash
cargo run -p playground
```

Run tests in a specific crate:

```bash
cargo test -p sorting
```

---

## Contributing

Some sections (e.g. `ml_basics`, `shared_utils`) are still empty - they will be filled step by step.
Feel free to open PRs with improvements, fixes, or new algorithms!
