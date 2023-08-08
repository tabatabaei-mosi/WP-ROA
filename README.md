# WP-ROA

The objective of this project mainly is to utilize ROA meta-heuristic algorithms for well placement optimization. This project is still under development by the authors.

## Notes

- All the codes must be run from the root directory of the project (`src`) to make the union of the paths work.
- Please when you add a new package, library, or module, add it to the `requirements.txt` file and follow the style.
- All the modules must be first tested before being added to the main code.
- Code must be well documented and commented to make it easy to understand and maintain.
- Please follow the [PEP 8](https://www.python.org/dev/peps/pep-0008/) style guide for Python code (clean code).
- Employ the modular programming paradigm to make the code more readable and maintainable.
- Clean code is a must.

## How to run the code

1. Install the requirements: `pip install -r requirements.txt`
2. Run the code with terminal: `python src/{main}.py`

## How to collaborate

1. Clone the repository: `git clone <repo_url>`
2. Create a new branch: `git checkout -b <branch_name>`
3. Make your changes
4. Commit your changes with an appropriate message: `git commit -m '<commit_message>'`
5. Push your changes to the new branch: `git push origin <branch_name>`
6. Submit a pull request
7. Wait for the pull request to be reviewed and merged
8. Delete your branch: `git branch -d <branch_name>`

## TODO

- [ ] Improve the Base ROA class
- [ ] Re-implement the TestEvaluation logging section (to include history of each model separately)
- [ ] Parallelization of Eclipse (Simulation process) and Python

