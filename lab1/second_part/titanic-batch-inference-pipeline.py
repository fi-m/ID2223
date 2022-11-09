###########     Initate modal    ################
import modal

stub = modal.Stub("lab-1-titanic")
image = (
    modal.Image.debian_slim()
    .apt_install(["libgomp1"])
    .pip_install(
        [
            "pandas",
            "numpy",
            "hopsworks",
            "scikit-learn",
            "seaborn",
        ]
    )
)
######################################################


@stub.function(image=image, secret=modal.Secret.from_name("titanic-secret"))
def main():
    import pandas as pd
    import hopsworks
    import joblib

    project = hopsworks.login()
    fs = project.get_feature_store()

    mr = project.get_model_registry()
    model = mr.get_model("titanic_model", version=2)
    model_dir = model.download()
    model = joblib.load(model_dir + "/titanic_model.pkl")

    fv = fs.get_feature_view(name="titanic_modal_view", version=1)

    batch_data = feature_view.get_batch_data()

    y_pred = model.predict(batch_data)
    
    # TODO

