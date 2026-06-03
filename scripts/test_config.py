from utils.config import load_config

config = load_config()
print("Config loaded successfully.")
print("Total PRB:", config["simulation"]["total_prb"])
print("Action values:", config["action"]["alpha_values"])