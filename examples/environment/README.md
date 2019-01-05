# Evironment

Included are sample environments for working with YeahML. These are the environments I use on my personal computer.

Until I have a more formal requirements.txt associated with the project, this should help some people start using yeahml.

Note: there are extra dependencies included in this environment that are not required for this project. This environment is only included to help people get started if they are unfamiliar with creating and managing environments.

## Creating the environment

### Select CPU or GPU environment

I cannot promise these will work for everyone, but they may help some people. Please use at your own risk.


### Create environment from selected file

See [conda documentation](https://conda.io/docs/user-guide/tasks/manage-environments.html#managing-environments)
```
conda create --name <env> --file <this file>
```

Example:
```
conda create --name yeahml --file cpu.yaml
source activate yeahml
... (later)
source deactivate
```