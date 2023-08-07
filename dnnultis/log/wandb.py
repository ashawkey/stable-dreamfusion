import shutil
import os
import subprocess
import wandb


class WandbUrls:
    def __init__(self, url):

        hash = url.split("/")[-2]
        project = url.split("/")[-3]
        entity = url.split("/")[-4]

        self.weight_url = url
        self.log_url = "https://app.wandb.ai/{}/{}/runs/{}/logs".format(entity, project, hash)
        self.chart_url = "https://app.wandb.ai/{}/{}/runs/{}".format(entity, project, hash)
        self.overview_url = "https://app.wandb.ai/{}/{}/runs/{}/overview".format(entity, project, hash)
        self.config_url = "https://app.wandb.ai/{}/{}/runs/{}/files/hydra-config.yaml".format(
            entity, project, hash
        )
        self.overrides_url = "https://app.wandb.ai/{}/{}/runs/{}/files/overrides.yaml".format(entity, project, hash)

    def __repr__(self):
        msg = "=================================================== WANDB URLS ===================================================================\n"
        for k, v in self.__dict__.items():
            msg += "{}: {}\n".format(k.upper(), v)
        msg += "=================================================================================================================================\n"
        return msg


class Wandb:
    IS_ACTIVE = False

    @staticmethod
    def set_urls_to_model(model, url):
        wandb_urls = WandbUrls(url)
        model.wandb = wandb_urls

    @staticmethod
    def _set_to_wandb_args(wandb_args, cfg, name):
        var = getattr(cfg.wandb, name, None)
        if var:
            wandb_args[name] = var

    @staticmethod
    def launch(cfg, launch: bool):
        if launch:

            Wandb.IS_ACTIVE = True

            wandb_args = {}
            wandb_args["resume"] = "allow"
            Wandb._set_to_wandb_args(wandb_args, cfg, "tags")
            Wandb._set_to_wandb_args(wandb_args, cfg, "project")
            Wandb._set_to_wandb_args(wandb_args, cfg, "name")
            Wandb._set_to_wandb_args(wandb_args, cfg, "entity")
            Wandb._set_to_wandb_args(wandb_args, cfg, "notes")
            Wandb._set_to_wandb_args(wandb_args, cfg, "config")
            Wandb._set_to_wandb_args(wandb_args, cfg, "id")

            try:
                commit_sha = subprocess.check_output(["git", "rev-parse", "HEAD"]).decode("ascii").strip()
                gitdiff = subprocess.check_output(["git", "diff", "--", "':!notebooks'"]).decode()
            except:
                commit_sha = "n/a"
                gitdiff = ""

            config = wandb_args.get("config", {})
            wandb_args["config"] = {
                **config,
                "run_path": os.getcwd(),
                "commit": commit_sha,
                "gitdiff": gitdiff
            }
            wandb.init(**wandb_args, sync_tensorboard=True)
            wandb.save(os.path.join(os.getcwd(), cfg.cfg_path))

    @staticmethod
    def add_file(file_path: str):
        if not Wandb.IS_ACTIVE:
            raise RuntimeError("wandb is inactive, please launch first.")

        filename = os.path.basename(file_path)
        shutil.copyfile(file_path, os.path.join(wandb.run.dir, filename))
