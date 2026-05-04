Authenticate a session on OCI `bmc_operator_access` tenancy using this command:

`oci session authenticate --profile-name DEFAULT --region us-chicago-1 --auth security_token`

Then activate the local environment and run the OCI smoke tests from the repo
root:

`conda activate cllm-env`

`python llm-script.py`

`python oci_model_panel_smoke_test.py`
