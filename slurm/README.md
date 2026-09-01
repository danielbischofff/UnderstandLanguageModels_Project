# Slurm jobs

Batch scripts for running experiments on the compute cluster.

Submit a steering YAML with:

```bash
sbatch slurm/steering.sbatch configs/steering/role_direction.yaml
```

To resume, pass an existing run directory as the second argument.
