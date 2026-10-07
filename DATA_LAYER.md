# Sreyaas data layer

Run `python data_layer.py` to rebuild the derived datasets. The script reads the source datasets but never writes to them.

## Generated files

- `data/structured_resumes.csv`: source resume fields plus standardized identity, education, context, experience, skills, and project fields. `detected_degrees` retains all detected credentials; `degree`/`primary_degree` select the highest supported degree. It also carries `course_final`, `skills_clean`, `skills_extracted`, `skills_clean_new`, `projects_extracted`, and `total_experience_years_final` aliases expected by `run_screening.py` when passed as `--base`.
- `data/structured_job_descriptions.csv`: standardized job schema, including explicit accepted degree/course fields. `required_degree`, `required_courses`, preferred experience, and preferred skills remain `NA` where the supplied JD does not distinguish them.
- `data/company_course_preference_mapping.csv`: existing company/JD preference mapping with explicit provenance, a `project_defined` status, and `unrelated_courses=NA` because no exhaustive unrelated-course list is supplied.
- `data/structured_counterfactuals.csv`: flattened counterfactual variants with `cf_id` plus an auditable `variant_json`. Only variants whose original attribute values match the source-derived structured resume are retained. The data is compatible with the counterfactual input shape in `run_screening.py`.
- `validation/structured_data_validation.csv`: validation results and legacy counterfactual rows excluded for review.

## Extraction policy and limits

Age is extracted only from an explicit `Age: N` style label. Dates, graduation years, skill durations, synthetic age/experience columns in `final/final_resumes.csv`, and inferred demographics are not used. Degree extraction detects all supported degree credentials and selects the highest supported level; `detected_degrees` preserves the distinct detected values. Course/specialization is extracted in the local span of that selected degree, preventing nearby diploma subjects and job titles from supplying the course. Experience uses explicit employment date ranges where employment wording supports them and unions overlapping intervals; a single explicit employment duration can be used when the text describes one work period, while multiple undated periods remain `NA` because overlap cannot be ruled out. Explicit overall statements such as `9 years of work experience` are accepted. Skill and project durations are excluded.

The existing college field is retained independently; an institution or education year is emitted only when a reliable institution/date parser exists (currently these are `NA`). Skills are separated at skill-duration markers, duration fragments and known heading artifacts are removed, and `.NET` plus comma-containing skill names are preserved. Project extraction requires an explicit project name/title/profile heading, stops at description/technology/responsibility labels, and returns repeated projects separately. Missing or ambiguous values remain `NA`.

Company names and course preferences come from `final/company_course_preferences.csv`; its provenance states that the mappings are project-defined from the existing JD context. This dataset does not establish a finer preferred/accepted/unrelated taxonomy than its preferred and accepted fields. Degree preferences stay in the JD dataset and are never treated as course preferences.

The existing audit and scoring logic is unchanged. To use structured resumes with the current scorer, pass `--base data/structured_resumes.csv`; the current standard counterfactual input remains compatible through the flattened `cf_id`/metadata columns. Do not substitute synthetic age/experience columns from `final/final_resumes.csv` as source evidence.
