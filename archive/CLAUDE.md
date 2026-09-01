Use modern Python typing -- e.g. dict, list instead of Dict, List.

Use uv whenever possible. 

Use pytest, and put all tests in /tests. Don't write test scripts at the root directory unless you intend to delete them shortly afterwards (and make sure you do so, keep a clean workspace). Any more permanent test should be written in /tests. 

Before you finalize your changes, please run pytest to ensure you haven't broken anything. Don't worry if tests fail that aren't related to what you've just done -- another Claude may have made some changes in parallel. Just tell me when this happens.