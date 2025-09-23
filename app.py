from flask import Flask, request, jsonify, send_file
import io
import matplotlib.pyplot as plt
import numpy as np
from scipy.stats import norm

app = Flask(__name__)

class ClimateAnomalyAnalyzer:
    def __init__(self, mu, sigma, X):
        self.mu = mu
        self.sigma = sigma
        self.X = X
        self.z_score = None
        self.p_less = None
        self.p_greater = None

    def compute_zscore(self):
        self.z_score = (self.X - self.mu) / self.sigma
        return self.z_score

    def compute_probabilities(self):
        if self.z_score is None:
            self.compute_zscore()
        self.p_less = norm.cdf(self.z_score)
        self.p_greater = 1 - self.p_less
        return self.p_less, self.p_greater

    def generate_plot(self):
        x = np.linspace(self.mu - 4*self.sigma, self.mu + 4*self.sigma, 1000)
        y = norm.pdf(x, self.mu, self.sigma)

        plt.figure(figsize=(8, 5))
        plt.plot(x, y, label="Normal Distribution", linewidth=2)

        x_fill = np.linspace(self.X, self.mu + 4*self.sigma, 500)
        y_fill = norm.pdf(x_fill, self.mu, self.sigma)
        plt.fill_between(x_fill, y_fill, alpha=0.5)

        plt.axvline(self.X, color="red", linestyle="--", label=f"X = {self.X}")
        plt.title("Probability of Temperature Anomaly > X")
        plt.xlabel("Temperature Anomaly (°C)")
        plt.ylabel("Density")
        plt.legend()

        buf = io.BytesIO()
        plt.savefig(buf, format='png')
        buf.seek(0)
        plt.close()
        return buf

@app.route('/analyze')
def analyze():
    try:
        mu = float(request.args.get('mu'))
        sigma = float(request.args.get('sigma'))
        X = float(request.args.get('X'))
    except (TypeError, ValueError):
        return jsonify({"error": "Invalid or missing parameters"}), 400

    analyzer = ClimateAnomalyAnalyzer(mu, sigma, X)
    z = analyzer.compute_zscore()
    p_less, p_greater = analyzer.compute_probabilities()

    return jsonify({
        "Z-score": round(z, 2),
        f"P(X ≤ {X})": round(p_less, 4),
        f"P(X > {X})": round(p_greater, 4),
        "plot_url": f"/plot?mu={mu}&sigma={sigma}&X={X}"
    })

@app.route('/plot')
def plot():
    try:
        mu = float(request.args.get('mu'))
        sigma = float(request.args.get('sigma'))
        X = float(request.args.get('X'))
    except (TypeError, ValueError):
        return "Invalid parameters", 400

    analyzer = ClimateAnomalyAnalyzer(mu, sigma, X)
    analyzer.compute_zscore()
    analyzer.compute_probabilities()
    plot_image = analyzer.generate_plot()
    return send_file(plot_image, mimetype='image/png')

if __name__ == '__main__':
    app.run(debug=True)
