# Evironment

Included is a sample environment for working with YeahML. This environment is the environment I use on my personal computer.

Until I have a more formal requirements.txt associated with the project, this should help some people start using yeahml.

Note: there are extra dependencies included in this environment that are not required for this project. This environment is only included to help people get started if they are unfamiliar with creating and managing environments.

## Creating the environment (CPU only, for now)
See [conda documentation](https://conda.io/docs/user-guide/tasks/manage-environments.html#creating-an-environment-from-an-environment-yml-file)
```
conda create --name <env> --file <this file>
```

Example:
```
conda create --name yeahml --file environment.yaml
source activate yeahml
...
source deactivate
```