import wandb

api = wandb.Api()

run = api.run("mldl_/FinalRotatedFemnist/xoan5e9s")
run.config["leftout"] = -1
run.update()