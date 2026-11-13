# 보안 정책 / Security policy

보안 취약점을 발견하셨다면 **공개 이슈 대신** 이 레포의 **Security → Report a vulnerability**(비공개 취약점 신고)로 알려 주세요. 확인 후 답변드리고, 고친 버전을 릴리스하면서 함께 안내합니다.

If you find a security issue, please **do not open a public issue**. Use **Security → Report a vulnerability** in this repository to report it privately. We will reply after checking and announce the fix with a release.

lora-mergecheck는 로컬 파일만 읽고 네트워크 요청을 보내지 않으며(`local_files_only`), `--json` 보고서와 `demo --out` 폴더에만 씁니다. / lora-mergecheck reads local files only, makes no network requests (`local_files_only`), and writes only the `--json` report and the `demo --out` folder.
