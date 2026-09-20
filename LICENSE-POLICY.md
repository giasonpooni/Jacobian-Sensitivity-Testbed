# License policy: JSPT

Status: **proposed transition; licensing counsel review is required before adoption**. This review-branch draft does not authorize changing the default-branch license, creating a release or tag, uploading a package, or replacing an existing distribution.

Proposed default: **MPL-2.0**. Classification: Public mathematical and sensitivity testbed. The complete unmodified standard license text is in `LICENSE`.

## Preserved MIT baseline

Default-branch review baseline: [`38c6912d96c18001c0fa83b030d707eaa491674d`](https://github.com/giasonpooni/Jacobian-Sensitivity-Propagation-Testbed/commit/38c6912d96c18001c0fa83b030d707eaa491674d). The published default branch remains MIT pending review.

The audited baseline is [`d910f5a1d7f6dd5f2dd87dfca66990f714f97b18`](https://github.com/giasonpooni/Jacobian-Sensitivity-Propagation-Testbed/commit/d910f5a1d7f6dd5f2dd87dfca66990f714f97b18), where the root license and package metadata declare MIT. Its original copyright and complete permission notice are copied byte-for-byte to `LICENSES/MIT-legacy.txt`.

Earlier copies received under MIT retain those terms; this proposal does not revoke, rewrite or retroactively replace that grant. Existing commits, tags, releases and distributed packages must remain intact. The legacy notice is retained as attribution and historical licensing evidence, not as an offer to apply MIT to all future additions.

The audit found no git tags or GitHub releases for this repository. This is an observation at the baseline, not a claim that no package or other copy has ever been distributed. Do not overwrite an existing package version under different terms; allocate an unused version before any package publication.

## Scope

After approval, MPL-2.0 is intended to cover project-owned source, tests, examples, build/configuration files and public documentation in this revision unless a file or directory carries separate terms. Documentation follows the source license; this proposal adds no separate Creative Commons grant. Mathematical facts and independent third-party rights are not relicensed by this policy.

Preserve all accurate existing copyright, patent, attribution and license notices. External dependencies keep their own licenses. Referenced or linked projects are not relicensed by changing this repository. A retained copy, generated record, third-party extract or vendor file must not be assigned new rights merely because it is present here.

## Review required before publication

- Confirm authority for project-owned contributions, including the relationship between Git author aliases and rights holders. Git authorship and automated co-author trailers are not proof of assignment or exclusive ownership.
- Confirm the scope and terms of any externally sourced material, preserve its notices, and record any exclusions before distribution.
- Review the complete standard license, its grants and obligations, and the proposed historical boundary. No custom changes are made to the standard license text.
- Keep package metadata, bundled license files and the README consistent, while preserving the provenance/version fields of historical scientific reports. No new physical-validation claim follows from this license proposal.

## Repository-specific boundaries

- Both Python `pyproject.toml` and the in-repository `rust/sensitivity-gate/Cargo.toml` use the proposed MPL-2.0 identifier. The NumPy oracle and Rust numerical gate remain in this repository; this is a license proposal, not a transfer of scientific ownership.
- NumPy, the optional JAX dependency and build/test tools retain their upstream terms. No vendored dependency license file or submodule was found in the audited tree. This is not a completed transitive dependency license audit.

## Standard text source

- https://www.mozilla.org/media/MPL/2.0/index.txt
- Retrieved standard-text SHA-256: `3f3d9e0024b1921b067d6f7f88deb4a60cbe7a78e76c64e3f1d7fc3b779b9d04`.
