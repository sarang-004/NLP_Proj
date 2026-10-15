# Evidence-based scoring methodology

## Versioning and compatibility

New scores use `evidence-v3`, defined in `screening/requirements.py` and
calculated by `run_screening.score()` for the legacy batch and audit path. Flask
upload screening uses `evidence-v4` for the skill category policy below. The configured component weights remain
skills 0.30, experience 0.20, education/course 0.15, projects 0.15, and
semantic similarity 0.20. No weight was tuned for the AWS Engineer example.

The previous scorer gave 1.0 for an unconfigured skills or experience factor,
compared total career years with every experience minimum, and assigned zero
to missing/unconfigured education data. It normalized only when project text
was absent. Those behaviors could create credit or penalties unsupported by
the JD/resume evidence.

In v3, the final score is the sum of each available component multiplied by
its effective normalized weight. Effective weights are the original configured
weights divided by the sum of the original weights for available components.
The following factors are unavailable and excluded from that denominator:

- A factor for which the JD specifies no criterion.
- A criterion whose candidate-side information is missing.
- Technology/activity-specific experience when the resume does not explicitly
  evidence the requested duration.
- Projects when no project text was extracted.

A present criterion not demonstrated in the resume remains a zero for that
criterion; it is described as “not demonstrated,” not as proof the candidate
does not have the skill. Missing candidate information is different and is
excluded. For the combined education/course component, only specified and
evidenced subcriteria are averaged; a missing subcriterion is left out. Degree
and course sub-evidence shares one component weight, split evenly among the
subcriteria actually included in that component.

V3 keeps the configured weights and available-factor aggregation formula. It
adds a contextual education rule for upload JDs that state a degree level and/or
subject. Degree title, level, subject, and specialization are retained
separately. For a preferred bachelor's criterion, a matching bachelor's level
and subject scores 1.0; a matching higher-level master's/integrated master's/
doctorate scores 0.8 as a subject match at a different level, not as an
equivalent bachelor's credential. An evidenced subject mismatch scores 0.0.
Missing degree or subject evidence is unavailable, not a mismatch. For a
mandatory bachelor's criterion, a master's credential alone does not prove the
bachelor's credential; the education score is unavailable pending review unless
a matching bachelor's credential is separately evidenced. If the JD omits the
subject, only the stated level is considered. A user-entered course requirement
remains a separate subcriterion. Legacy structured jobs without contextual
education criteria retain their established title-list matcher.

The previous upload behavior expanded a generic preferred bachelor's phrase
into a title preference/accepted-title list and compared degree strings without
using the JD's subject as a separate signal. A higher credential with a matching
subject could therefore receive 0.0 solely because its title was M.Tech. V3
replaces that upload-specific decision with the contextual rule above; it does
not change the configured 0.15 education/course weight or the final-score
normalization formula.

## Requirement and evidence rules

`screening.requirements.classify_jd_requirements()` preserves each source
statement and classifies it as an explicit minimum, preferred, responsibility,
general mention, or uncertain. Qualifications sections, explicit words such
as “required,” numeric durations, and global flexibility wording affect these
labels. Responsibility statements are retained in context and excluded from
skill requirements. An uncertain criterion is not silently promoted into a
scored requirement. The extractor uses the project's structured JD skill
vocabulary plus a common technology list; unknown phrasing may need recruiter
review.

Skill component evidence has reproducible categories:

- Professional use evidenced in an action statement: 1.0.
- Exact skill listed without a supporting work statement: 0.5.
- Existing declared alias/related match: 0.5, labeled related/transferable.
- Not demonstrated in supplied resume: 0.0.

These are transparent heuristic evidence levels, not validated measures of
proficiency. Repeated mentions do not add credit. The explanation retains the
JD clause and a resume excerpt when available.

## Upload screening v4: required and preferred skills

Within the upload screening skills factor, explicit minimum/required criteria
have criterion weight 1.0 and preferred criteria have criterion weight 0.5.
Their evidence credits remain unchanged. The factor score is the weighted mean
of evidence credits; the overall configured skills weight remains 0.30 and the
outer available-factor normalization is unchanged. General mentions,
responsibility statements, and uncertain criteria remain context, not scored
skill criteria. This 2:1 policy is a transparent product choice, not an
empirically calibrated hiring rule. The API exposes each criterion weight,
evidence credit, and the numerator/denominator used for the skill score.

Only the Flask upload-screening path opts into this policy. The v3 audit and
legacy batch path continue using equal per-skill averaging. No audit module or
validated audit output was changed or regenerated. New upload scores are tagged
`evidence-v4`; callers must not compare them as same-methodology values with
v3 scores or audit deltas.

Evidence coverage is reported as the share of scored criteria with extracted
resume evidence. It is not a fit or quality score. A missing mention is labeled
not demonstrated, not proof that a candidate lacks the qualification.
Education mismatches are labeled explicitly only when the extracted subject or
level conflicts with the stated criterion. Project and semantic similarities
are aggregate cosine signals with supporting extracted text shown where
available; neither proves a specific qualification or identifies a causal
phrase.

An unqualified numeric total-experience requirement is compared against
explicit total years. Employment periods accept named-month, year-only, and
`MM.YYYY` date ranges, are unioned to avoid double-counting overlaps, and use
the recorded `experience_as_of` date for open “now/present” periods. Resume
profile-stated years are retained separately from date-derived total years.
If an experience requirement names a technology, the
duration must be tied to that technology and requested activity in resume
evidence. A project duration associated with a technology/tools list and
related activity is retained as contextual evidence requiring review, but is
not treated as technology-specific years. Existing employment-date parsing
continues to union overlapping periods for total experience; it does not
convert total tenure into technology tenure.

Semantic similarity remains the configured MiniLM cosine score mapped from
[-1, 1] to [0, 1]. Project similarity uses the same calculation over extracted
project text. These are ranking signals, not causal explanations.

## Counterfactual and legacy outputs

New base and counterfactual scores are tagged with the same methodology
version, and the pipeline rejects an in-memory base/variant mismatch. The
delta and case-reconciliation utilities reject known cross-version joins.
Existing validated output files were not regenerated or relabeled. They have
no trustworthy methodology-version field and must be treated as
`legacy_unversioned`; their historical deltas do not correspond to v2 unless
they are rescored through an explicit controlled run. Score changes are not
proof of discrimination.

## Comparability

V2 scores are reproducible from the stored raw component scores, effective
weights, and weighted contributions. Because unavailable factors are excluded,
two candidates may have different effective weights when their evidence
availability differs. Review the factor evidence and denominator along with
the final score. The method has deterministic tests, including the supplied
AWS Engineer and Mobile Application Testing PDFs when those files are present;
this does not establish predictive validity or hiring suitability.
