//! Builds of one annotated WGSL source.
//!
//! WGSL has no preprocessor, so a kernel that exists in several builds (an
//! RGBA or a two-plane YUV source, with or without the per-row pass, …) used
//! to be one file per build, each a copy of the others with a few lines
//! changed. Here the source is ONE file and the differences are marked with
//! line directives, which are ordinary comments to a WGSL reader:
//!
//! ```text
//! //#if NAME        the lines up to the matching //#else or //#endif are
//! //#if !NAME       kept when the flag is set (or, with `!`, when it is not)
//! //#else
//! //#endif
//! //#define NAME    names the lines up to //#enddef; they are not emitted
//! //#enddef         where they are defined
//! //#use NAME       emits a defined block here, composed with the same flags
//! ```
//!
//! Conditionals nest. A directive must sit on a line of its own.

/// Compose `src` with `flags` set. Panics on a malformed source (an unknown
/// directive, an unbalanced conditional, an undefined block): the sources
/// are compiled in, so that is a programming error, and the tests compose
/// every build.
pub(crate) fn compose(src: &str, flags: &[&str]) -> String {
    let mut blocks: Vec<(&str, Vec<&str>)> = Vec::new();
    let mut body: Vec<&str> = Vec::new();
    let mut defining: Option<(&str, Vec<&str>)> = None;
    for line in src.lines() {
        let directive = line.trim();
        match &mut defining {
            Some((name, lines)) => {
                if directive == "//#enddef" {
                    blocks.push((name, std::mem::take(lines)));
                    defining = None;
                } else {
                    lines.push(line);
                }
            }
            None => match directive.strip_prefix("//#define ") {
                Some(name) => defining = Some((name.trim(), Vec::new())),
                None => body.push(line),
            },
        }
    }
    assert!(defining.is_none(), "shader source: //#define without //#enddef");
    let mut out = String::with_capacity(src.len());
    emit(&body, &blocks, flags, &mut out, 0);
    out
}

fn emit(lines: &[&str], blocks: &[(&str, Vec<&str>)], flags: &[&str], out: &mut String, depth: usize) {
    assert!(depth < 8, "shader source: //#use nests too deep (a block that uses itself?)");
    // One entry per open conditional: is its current branch kept?
    let mut kept: Vec<bool> = Vec::new();
    for line in lines {
        let directive = line.trim();
        if let Some(cond) = directive.strip_prefix("//#if ") {
            let cond = cond.trim();
            let (name, wanted) = match cond.strip_prefix('!') {
                Some(name) => (name, false),
                None => (cond, true),
            };
            kept.push(flags.contains(&name) == wanted);
        } else if directive == "//#else" {
            let branch = kept.last_mut().expect("shader source: //#else without //#if");
            *branch = !*branch;
        } else if directive == "//#endif" {
            kept.pop().expect("shader source: //#endif without //#if");
        } else if kept.iter().all(|k| *k) {
            if let Some(name) = directive.strip_prefix("//#use ") {
                let block = blocks.iter().find(|(n, _)| *n == name.trim())
                    .unwrap_or_else(|| panic!("shader source: //#use of an undefined block {name:?}"));
                emit(&block.1, blocks, flags, out, depth + 1);
            } else {
                assert!(!directive.starts_with("//#"), "shader source: unknown directive {directive:?}");
                out.push_str(line);
                out.push('\n');
            }
        }
    }
    assert!(kept.is_empty(), "shader source: //#if without //#endif");
}

#[cfg(test)]
mod tests {
    use super::compose;

    const SRC: &str = "a\n//#if X\nx\n//#if !Y\nx-not-y\n//#else\nx-and-y\n//#endif\n//#else\nnot-x\n//#endif\n\
//#define BODY\nbody\n//#if Y\nbody-y\n//#endif\n//#enddef\n  //#use BODY\nz\n";

    #[test]
    fn conditionals_nest_and_blocks_follow_the_flags() {
        assert_eq!(compose(SRC, &[]), "a\nnot-x\nbody\nz\n");
        assert_eq!(compose(SRC, &["X"]), "a\nx\nx-not-y\nbody\nz\n");
        assert_eq!(compose(SRC, &["X", "Y"]), "a\nx\nx-and-y\nbody\nbody-y\nz\n");
        assert_eq!(compose(SRC, &["Y"]), "a\nnot-x\nbody\nbody-y\nz\n");
        // A block used inside a branch that is dropped is not emitted.
        assert_eq!(compose("//#define B\nb\n//#enddef\n//#if X\n//#use B\n//#endif\nend\n", &[]), "end\n");
    }

    #[test]
    #[should_panic(expected = "unknown directive")]
    fn an_unknown_directive_is_an_error() {
        compose("//#elif X\n", &[]);
    }

    #[test]
    #[should_panic(expected = "without //#endif")]
    fn an_open_conditional_is_an_error() {
        compose("//#if X\nx\n", &["X"]);
    }
}
