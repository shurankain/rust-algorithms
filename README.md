# Rust Algorithms

Modular Rust workspace containing implementations of classic algorithms across categories. The primary goal is to **learn, understand, and document algorithmic approaches** while deepening knowledge of the Rust programming language.

---

## Structure

Uses Cargo workspaces - to isolate infrastructure by topics

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
|- ml_basics/
|- shared_utils/           # commons structures (Graph, Matrix e.g.)
|- playground/             # binary for examples execution
```

## Usage

To run or test a specific algorithm, use the `playground` binary:

```bash
cargo run -p playground
```

To run tests in a specific crate:

```bash
cargo test -p sorting
```
