# decision workbench

enterprise multimodal intent router and multi-model benchmark for gemini enterprise.

## architecture

evaluates queries concurrently across 9 inference models to route across 11 enterprise destinations:

### models
- `gemini-3.8-flash`: multimodal image/video understanding and ocr intent extraction
- `typesafe-jev`: deterministic token verification and semantic safety boundaries
- `convai-laya`: modernbert-large bidirectional non-autoregressive encoder
- `bigquery-ml`: warehouse boosted tree classifier (zero data egress)
- `catboost`: symmetric oblivious decision trees
- `xgboost`: exact greedy split tree ensemble with native svg explainability
- `gradientxgb`: lightgbm histogram gradient-boosted decision trees
- `scikit-learn`: multinomial logistic regression with platt scaling
- `statsmodels`: maximum likelihood multinomial logit parameter estimation

### destinations
1. `training_search`: lms modules, onboarding, food prep and equipment cleaning sops
2. `servicenow_article_search`: knowledge base troubleshooting and break-fix guides
3. `servicenow_form_search`: service catalog request forms
4. `servicenow_case_search`: incident status, case telemetry, and sla logs
5. `intranet_home_search`: corporate governance, policies, and benefits
6. `sharepoint_doc_search`: spreadsheets, slide decks, and team files
7. `confluence_wiki_search`: engineering architecture specs, design docs, and runbooks
8. `jira_issue_search`: user stories, sprint backlog, and bug defects
9. `github_code_search`: git repositories, pull requests, and ci/cd workflows
10. `artifactory_package_search`: container registries, docker images, and packages
11. `tableau_report_search`: bi dashboards and kpi metric summaries

## interfaces

- `a2a json-rpc 2.0`: streaming task lifecycle endpoint for gemini enterprise (`message/send`)
- `a2ui v0.8 / v0.9`: native interactive tabs, decision tree svgs, feature gain charts, image and video players
- `http rest`: `/healthz`, `/api/compare`, and interactive companion workbench

## deployment

```bash
./deploy.sh
```

endpoint: `https://gemini-jev-decision-workbench-36231825761.us-central1.run.app`
agent discovery: `/.well-known/agent-card.json`
