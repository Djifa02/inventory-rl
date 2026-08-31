import gradio as gr
import numpy as np
import pandas as pd

from agent.ddpg import DDPGAgent


# ==================================================
# PARAMÈTRES
# ==================================================

MODEL_PATH = "results/models/ddpg_final"

MAX_INVENTORY = 1000.0
MAX_ORDER = 200.0


# ==================================================
# DONNÉES
# ==================================================

df = pd.read_csv("data/retail_store_inventory.csv")

DEMAND_MIN = df["Demand Forecast"].min()
DEMAND_MAX = df["Demand Forecast"].max()


# ==================================================
# CHARGEMENT DU MODÈLE
# ==================================================

def load_agent():

    agent = DDPGAgent(
        state_dim=2,
        action_dim=1,
        action_low=0.0,
        action_high=MAX_ORDER,
    )

    agent.load(MODEL_PATH)

    return agent


agent = load_agent()


# ==================================================
# PRÉDICTION
# ==================================================

def predict_order(stock, demand_forecast):

    try:

        if stock is None or demand_forecast is None:
            return None, "Veuillez renseigner les deux valeurs."

        if stock < 0:
            return None, "Le stock ne peut pas être négatif."

        if demand_forecast < 0:
            return None, "La demande prévue ne peut pas être négative."

        if stock > MAX_INVENTORY:
            return None, f"Le stock doit être compris entre 0 et {MAX_INVENTORY:.0f}."

        # ------------------------------------------
        # Normalisation du stock
        # ------------------------------------------

        stock_normalized = stock / MAX_INVENTORY

        stock_normalized = np.clip(
            stock_normalized,
            0.0,
            1.0
        )

        # ------------------------------------------
        # Normalisation de la demande
        # ------------------------------------------

        demand_normalized = (
            demand_forecast - DEMAND_MIN
        ) / (
            DEMAND_MAX - DEMAND_MIN + 1e-6
        )

        demand_normalized = np.clip(
            demand_normalized,
            0.0,
            1.0
        )

        # ------------------------------------------
        # État du modèle
        # ------------------------------------------

        state = np.array(
            [
                stock_normalized,
                demand_normalized
            ],
            dtype=np.float32
        )

        # ------------------------------------------
        # Prédiction DDPG
        # ------------------------------------------

        action = agent.predict(
            state,
            deterministic=True
        )

        quantity = float(
            np.asarray(action).flatten()[0]
        )

        quantity = np.clip(
            quantity,
            0.0,
            MAX_ORDER
        )

        quantity = round(quantity, 2)

        # ------------------------------------------
        # Message explicatif
        # ------------------------------------------

        if quantity == 0:
            message = (
                "Le modèle ne recommande pas de réapprovisionnement "
                "pour cet état."
            )

        elif quantity >= MAX_ORDER:
            message = (
                "Le modèle recommande la quantité maximale "
                "de réapprovisionnement autorisée."
            )

        else:
            message = (
                f"Le modèle recommande de commander "
                f"{quantity:.2f} unité(s)."
            )

        return quantity, message

    except Exception as e:

        return None, f"Erreur : {str(e)}"


# ==================================================
# INTERFACE GRADIO
# ==================================================

with gr.Blocks(
    title="Inventory-RL"
) as demo:

    gr.Markdown(
        """
        # 📦 Inventory-RL

        ## Optimisation de la gestion des stocks

        Cette application utilise un agent **DDPG (Deep Deterministic
        Policy Gradient)** pour recommander une quantité de
        réapprovisionnement à partir du **stock actuel** et de la
        **demande prévue**.

        ---
        """
    )

    with gr.Row():

        with gr.Column():

            stock = gr.Number(
                label="📦 Stock actuel",
                value=50,
                minimum=0,
                maximum=1000,
                precision=2,
            )

            demand = gr.Number(
                label="📈 Demande prévue",
                value=30,
                minimum=0,
                precision=2,
            )

            with gr.Row():

                predict_button = gr.Button(
                    "🔮 Prédire",
                    variant="primary",
                )

                reset_button = gr.Button(
                    "↩️ Réinitialiser",
                )

        with gr.Column():

            result = gr.Number(
                label="📦 Quantité recommandée",
                interactive=False,
                precision=2,
            )

            explanation = gr.Markdown(
                "Entrez les données puis cliquez sur **Prédire**."
            )

    gr.Markdown(
        """
        ---

        ### ℹ️ Informations sur le modèle

        - **Algorithme :** DDPG
        - **État utilisé :** stock actuel + demande prévue
        - **Action :** quantité à commander
        - **Quantité maximale autorisée :** 200 unités
        """
    )

    # ----------------------------------------------
    # Bouton prédiction
    # ----------------------------------------------

    predict_button.click(
        fn=predict_order,
        inputs=[
            stock,
            demand
        ],
        outputs=[
            result,
            explanation
        ],
    )

    # ----------------------------------------------
    # Bouton réinitialisation
    # ----------------------------------------------

    reset_button.click(
        fn=lambda: (
            50,
            30,
            None,
            "Entrez les données puis cliquez sur **Prédire**."
        ),
        inputs=[],
        outputs=[
            stock,
            demand,
            result,
            explanation
        ],
    )


# ==================================================
# LANCEMENT
# ==================================================

if __name__ == "__main__":

    demo.launch()