//! Compile-fail tests using trybuild.
//! Verifies that invalid dimension/frame combinations produce compile errors.

#[test]
fn compile_fail_tests() {
    let t = trybuild::TestCases::new();
    t.compile_fail("tests/compile_fail/*.rs");
}
