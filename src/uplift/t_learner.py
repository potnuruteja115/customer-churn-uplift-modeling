from sklearn.base import clone


class TLearner:
    def __init__(self, control_model, treatment_model, clone_models=True):

        if clone_models:
            self.control_model = clone(control_model)
            self.treatment_model = clone(treatment_model)
        else:
            self.control_model = control_model
            self.treatment_model = treatment_model

    def fit(self, X, treatment, y):

        control_mask = treatment == 0
        treatment_mask = treatment == 1

        if control_mask.sum() == 0:
            raise ValueError("No control samples found.")

        if treatment_mask.sum() == 0:
            raise ValueError("No treatment samples found.")

        self.control_model.fit(
            X[control_mask],
            y[control_mask]
        )

        self.treatment_model.fit(
            X[treatment_mask],
            y[treatment_mask]
        )

        return self

    def predict_uplift(self, X):

        treatment_probability = (
            self.treatment_model.predict_proba(X)[:, 1]
        )

        control_probability = (
            self.control_model.predict_proba(X)[:, 1]
        )

        return treatment_probability - control_probability

    def predict_proba_control(self, X):

        return self.control_model.predict_proba(X)[:, 1]

    def predict_proba_treatment(self, X):

        return self.treatment_model.predict_proba(X)[:, 1]