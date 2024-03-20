# How to make contribution

1. Fork this repo to your github account

2. Clone your fork

3. Create a feature branch

4. Make changes, including but not limited to create new model, bug fix, documentation, tutorials, etc.

5. Pre-commit check and push, we use clang-format to do coding style checking, and the coding style is following google c++ coding style with 4-space.

```
pip install pre-commit
pip install clang-format

cd tensorrtx/
git add [files-to-commit]
pre-commit run

# fix pre-commit errors, then git add files-to-commit again
git add [files-to-commit]

git commit -m "describe your commit"

git push origin [feature-branch]
```

6. Submit a pull-request on github web UI to master branch of wang-xinyu/tensorrtx.
