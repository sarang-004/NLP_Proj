
import os, re, ast, argparse
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "sentence-transformers/all-MiniLM-L6-v2"

# Final score = 100%
WEIGHTS = {
    "skills": 0.30,
    "experience": 0.20,
    "education_course": 0.15,
    "projects": 0.15,
    "semantic": 0.20,
}

def clean(x):
    return "" if pd.isna(x) else str(x).strip()

def norm(x):
    x = clean(x).lower().replace("&", " and ")
    x = re.sub(r"[\u2010-\u2015]", "-", x)
    x = re.sub(r"[^a-z0-9+#./ -]", " ", x)
    return re.sub(r"\s+", " ", x).strip()

def num(x, default=0.0):
    try:
        if pd.isna(x): return default
        return float(x)
    except Exception:
        return default

def parse_list(x):
    if isinstance(x, (list, tuple, set)):
        return [clean(v) for v in x if clean(v)]
    raw = clean(x)
    if not raw or raw in {"[]", "None", "nan"}:
        return []
    try:
        obj = ast.literal_eval(raw)
        if isinstance(obj, (list, tuple, set)):
            return [clean(v) for v in obj if clean(v)]
    except Exception:
        pass
    raw = raw.strip("[]")
    parts = re.split(r"\s*\|\s*|\s*;\s*|\s*,\s*", raw)
    return [p.strip(" '\"") for p in parts if p.strip(" '\"")]

def norm_skill(x):
    x = norm(x)
    x = re.sub(r"\s*-\s*(?:exprience|experience).*?$", "", x)
    x = re.sub(r"\s+\d+(?:\.\d+)?\s*months?\b", "", x)
    return re.sub(r"\s+", " ", x).strip()

def resume_skills(row):
    for col in ["skills_clean_new", "skills_extracted", "skills_clean"]:
        vals = parse_list(row.get(col, ""))
        if vals:
            return {norm_skill(v) for v in vals if norm_skill(v)}
    return set()

def required_skills(job):
    return {norm_skill(x) for x in clean(job["required_skills"]).split("|") if norm_skill(x)}

ALIASES = {
    "sql": {"sql", "mysql"},
    "mysql": {"mysql", "sql"},
    "databases": {"database", "databases", "mysql", "oracle"},
    "database": {"database", "databases", "mysql", "oracle"},
    "javascript": {"javascript", "javascript/typescript"},
    "html": {"html", "html5"},
    "css": {"css", "css3"},
    "networking": {"networking", "computer networking"},
    "automation": {"automation", "test automation", "automation testing"},
}

def skill_match(req, have):
    return req == have or have in ALIASES.get(req, set())

def score_skills(row, job):
    req = required_skills(job)
    if not req: return 1.0
    have = resume_skills(row)
    return sum(any(skill_match(r,h) for h in have) for r in req) / len(req)

def experience_years(row):
    y = num(row.get("total_experience_years_final", np.nan), np.nan)
    if not np.isnan(y): return max(0.0, y)
    m = num(row.get("total_experience_months_final", np.nan), np.nan)
    if not np.isnan(m): return max(0.0, m/12.0)
    return max(0.0, num(row.get("total_experience_years", 0)))

def score_experience(row, job):
    minimum = num(job["minimum_total_experience_years"])
    return 1.0 if minimum <= 0 else float(np.clip(experience_years(row)/minimum, 0, 1))

def score_education(row, job):
    degree, course = norm(row.get("primary_degree","")), norm(row.get("course_final",""))
    ad = {norm(x) for x in clean(job["accepted_degrees"]).split("|") if norm(x)}
    ac = {norm(x) for x in clean(job["accepted_courses"]).split("|") if norm(x)}
    pdg, pc = norm(job["preferred_degree"]), norm(job["preferred_course"])

    def one(v, pref, accepted):
        if not v: return 0.0
        if v == pref: return 1.0
        if v in accepted: return 0.8
        if any(a and (a in v or v in a) for a in accepted): return 0.8
        return 0.0

    return (one(degree,pdg,ad) + one(course,pc,ac))/2

def project_text(row):
    vals = parse_list(row.get("projects_extracted",""))
    if vals: return " ".join(vals)
    # Fallback: extract explicit project-title blocks from resume text.
    text = clean(row.get("resume_text",""))
    found=[]
    for m in re.finditer(
        r"(?:project title|project)\s*[:\-]\s*(.*?)(?="
        r"\brole\s*:|\btools and technologies\s*:|\bproject title\s*:|"
        r"\beducation\b|\bskills?\b|$)", text, re.I|re.S):
        found.append(m.group(1).strip())
    return " ".join(found)

def jd_text(job):
    return (
        f"Job title: {clean(job['job_title'])}. "
        f"Required skills: {clean(job['required_skills'])}. "
        f"Preferred degree: {clean(job['preferred_degree'])}. "
        f"Accepted degrees: {clean(job['accepted_degrees'])}. "
        f"Preferred course: {clean(job['preferred_course'])}. "
        f"Accepted courses: {clean(job['accepted_courses'])}. "
        f"Minimum experience: {clean(job['minimum_total_experience_years'])} years."
    )

def cosine01(a,b,model):
    if not clean(a) or not clean(b): return 0.0
    e = model.encode([clean(a),clean(b)], normalize_embeddings=True,
                     convert_to_numpy=True, show_progress_bar=False)
    c = float(cosine_similarity([e[0]],[e[1]])[0,0])
    return float(np.clip((c+1)/2,0,1))

def patch_text(base_row, cfrow):
    text = clean(base_row.get("resume_text",""))
    old = clean(cfrow.get("original_value",""))
    new = clean(cfrow.get("counterfactual_value",""))
    attrs = [a.strip() for a in clean(cfrow.get("changed_attribute","")).split("+") if a.strip()]
    oldv = [x.strip() for x in old.split("|")] if len(attrs)>1 else [old]
    newv = [x.strip() for x in new.split("|")] if len(attrs)>1 else [new]
    for i, attr in enumerate(attrs):
        o = oldv[i] if i < len(oldv) else old
        v = newv[i] if i < len(newv) else new
        if o and v:
            text = re.sub(re.escape(o), v, text, flags=re.I)
    # Explicitly expose the changed proxy even if it wasn't present in text.
    meta=[]
    for i, attr in enumerate(attrs):
        v = newv[i] if i < len(newv) else new
        meta.append(f"{attr}: {v}")
    return text + ("\nCounterfactual attributes: " + "; ".join(meta) if meta else "")

def apply_cf(base_row, cfrow):
    row = base_row.copy()
    attrs = [a.strip() for a in clean(cfrow["changed_attribute"]).split("+") if a.strip()]
    vals = [x.strip() for x in clean(cfrow["counterfactual_value"]).split("|")] if len(attrs)>1 else [clean(cfrow["counterfactual_value"])]
    for i, attr in enumerate(attrs):
        v = vals[i] if i < len(vals) else vals[0]
        if attr == "name": row["name"] = v
        elif attr == "college": row["college"] = v
        elif attr == "city": row["city"] = v
        elif attr == "age": row["age_final"] = num(v, row.get("age_final",0))
        elif attr == "experience":
            months = num(v, row.get("total_experience_months_final",0))
            row["total_experience_months_final"] = months
            row["total_experience_years_final"] = months/12.0
        elif attr == "degree": row["primary_degree"] = v
        elif attr == "course": row["course_final"] = v
    row["_patched_text"] = patch_text(base_row, cfrow)
    return row

def score(row, job, model):
    sem = cosine01(row.get("_patched_text",row.get("resume_text","")), jd_text(job), model)
    proj_text = project_text(row)
    proj = cosine01(proj_text, jd_text(job), model) if proj_text else 0.0
    sk = score_skills(row,job)
    ex = score_experience(row,job)
    ed = score_education(row,job)

    # Normalize weights over available components so missing project extraction
    # does not automatically penalize a candidate.
    values = {"skills":sk,"experience":ex,"education_course":ed,"projects":proj,"semantic":sem}
    available = {k:v for k,v in values.items() if not (k=="projects" and not proj_text)}
    denom = sum(WEIGHTS[k] for k in available)
    final = sum(WEIGHTS[k]*v for k,v in available.items()) / denom

    return {
        "skills_score": round(sk,6),
        "experience_score": round(ex,6),
        "education_course_score": round(ed,6),
        "projects_score": round(proj,6),
        "semantic_score": round(sem,6),
        "final_score": round(float(np.clip(final,0,1)),6)
    }

def main():
    ap=argparse.ArgumentParser()
    ap.add_argument("--base",default="final_resumes.csv")
    ap.add_argument("--jobs",default="job_requirements_complete.csv")
    ap.add_argument("--counterfactual",default="counterfactual_resumes_valid.csv")
    ap.add_argument("--outdir",default="screening_output_final")
    args=ap.parse_args()
    os.makedirs(args.outdir,exist_ok=True)

    base=pd.read_csv(args.base)
    jobs=pd.read_csv(args.jobs)
    cf=pd.read_csv(args.counterfactual)

    jobmap=jobs.set_index("job_id").to_dict("index")
    base["job_id"]=base["source_category"].map(
        dict(zip(jobs["source_category"],jobs["job_id"]))
    )

    # All 25 should now map.
    unmapped=base[base["job_id"].isna()][["resume_id","source_category"]].drop_duplicates()
    unmapped.to_csv(os.path.join(args.outdir,"unmapped_resume_categories.csv"),index=False)

    model=SentenceTransformer(MODEL_NAME)
    supported=base[base["job_id"].notna()].copy()
    lookup={r["resume_id"]:r for _,r in supported.iterrows()}

    base_scores={}
    base_details={}
    print(f"Scoring {len(supported)} base resumes...")
    for rid,row in lookup.items():
        r=score(row,jobmap[row["job_id"]],model)
        base_scores[rid]=r["final_score"]
        base_details[rid]=r

    cf2=cf.merge(
        supported[["resume_id","job_id"]],
        left_on="original_resume_id",
        right_on="resume_id",
        how="inner",
        suffixes=("","_base")
    )

    main_rows=[]
    detailed=[]
    print(f"Scoring {len(cf2)} counterfactual variants...")
    for _,c in cf2.iterrows():
        rid=c["original_resume_id"]
        row=apply_cf(lookup[rid],c)
        r=score(row,jobmap[c["job_id"]],model)
        b=base_details[rid]
        bs=base_scores[rid]
        main_rows.append({
            "resume_id":rid,
            "job_id":c["job_id"],
            "variant_id":clean(c["cf_id"]),
            "proxy_type":clean(c["changed_attribute"]),
            "proxy_value":clean(c["counterfactual_value"]),
            "base_score":bs,
            "variant_score":r["final_score"],
        })
        detailed.append({
            "resume_id":rid,"job_id":c["job_id"],
            "variant_id":clean(c["cf_id"]),
            "proxy_type":clean(c["changed_attribute"]),
            "proxy_value":clean(c["counterfactual_value"]),
            "base_skills_score":b["skills_score"],
            "base_experience_score":b["experience_score"],
            "base_education_course_score":b["education_course_score"],
            "base_projects_score":b["projects_score"],
            "base_semantic_score":b["semantic_score"],
            "base_score":bs,
            "variant_skills_score":r["skills_score"],
            "variant_experience_score":r["experience_score"],
            "variant_education_course_score":r["education_course_score"],
            "variant_projects_score":r["projects_score"],
            "variant_semantic_score":r["semantic_score"],
            "variant_score":r["final_score"],
            "score_delta":round(r["final_score"]-bs,6),
        })

    out=pd.DataFrame(main_rows)
    det=pd.DataFrame(detailed)
    out.to_csv(os.path.join(args.outdir,"screening_results.csv"),index=False)
    det.to_csv(os.path.join(args.outdir,"screening_results_detailed.csv"),index=False)

    mapping=jobs[["job_id","source_category","job_title"]].copy()
    mapping.to_csv(os.path.join(args.outdir,"role_to_job_mapping.csv"),index=False)

    validation=pd.DataFrame([{
        "base_resumes_total":len(base),
        "base_resumes_scored":len(supported),
        "base_resumes_unmapped":len(unmapped),
        "counterfactual_total":len(cf),
        "counterfactuals_scored":len(out),
        "duplicate_variant_ids":int(out["variant_id"].duplicated().sum()) if len(out) else 0,
        "max_base_scores_per_resume":int(out.groupby("resume_id")["base_score"].nunique().max()) if len(out) else 0,
        "max_job_ids_per_resume":int(out.groupby("resume_id")["job_id"].nunique().max()) if len(out) else 0,
        "score_delta_min":float(det["score_delta"].min()) if len(det) else 0,
        "score_delta_max":float(det["score_delta"].max()) if len(det) else 0,
    }])
    validation.to_csv(os.path.join(args.outdir,"validation_report.csv"),index=False)

    # Proxy summary for quick bias-audit inspection.
    if len(det):
        summary=det.groupby("proxy_type").agg(
            variants=("variant_id","count"),
            mean_delta=("score_delta","mean"),
            median_delta=("score_delta","median"),
            min_delta=("score_delta","min"),
            max_delta=("score_delta","max"),
            mean_abs_delta=("score_delta",lambda x: np.mean(np.abs(x)))
        ).reset_index()
    else:
        summary=pd.DataFrame()
    summary.to_csv(os.path.join(args.outdir,"proxy_delta_summary.csv"),index=False)

    print("\nCOMPLETE")
    print(validation.to_string(index=False))
    print(f"\nOutput: {args.outdir}")

if __name__=="__main__":
    main()
