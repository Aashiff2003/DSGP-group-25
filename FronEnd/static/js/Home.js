document.addEventListener("DOMContentLoaded", function () {
    const ctx = document.getElementById("riskChart").getContext("2d");

    // Gradient fill for the chart
    let gradient = ctx.createLinearGradient(0, 0, 0, 400);
    gradient.addColorStop(0, "rgba(92, 73, 101, 0.7)");
    gradient.addColorStop(1, "rgba(92, 73, 101, 0.2)");

    // Chart configuration
    const riskChart = new Chart(ctx, {
        type: "line",
        data: {
            labels: [],
            datasets: [
                {
                    label: "Risk Level",
                    data: [],
                    borderColor: "#5c4965", // Line color
                    borderWidth: 3,
                    tension: 0.4, // Smooth curves
                    backgroundColor: gradient, // Apply gradient fill
                    fill: true, // Enable fill under line
                    pointBackgroundColor: "#5c4965", // Point color
                    pointBorderColor: "#ffffff", // White border for points
                    pointRadius: 5, // Size of points
                    pointHoverRadius: 7, // Size when hovered
                    pointHoverBorderWidth: 2,
                    pointHoverBorderColor: "rgba(255,255,255,0.8)",
                },
            ],
        },
        options: {
            responsive: true,
            maintainAspectRatio: false,
            scales: {
                x: {
                    type: "time",
                    time: {
                        unit: "second",
                        tooltipFormat: "HH:mm:ss",
                    },
                    grid: { display: false },

                },
                y: {
                    display: true,
                    grid: { display: false },

                },
            },
            plugins: {
                legend: {
                    display: false,
                },
                tooltip: {
                    enabled: true,
                    backgroundColor: "#2c3e50",
                    titleFont: { size: 14 },
                    bodyFont: { size: 12 },
                    padding: 10,
                    cornerRadius: 8,
                },
            },
            animation: {
                duration: 500,
            },
        },
    });

    // Function to update the chart every second
    function updateChart() {
        fetch("/stats")  // Make sure this is the correct Flask API endpoint
            .then(response => response.json())
            .then(data => {
                const now = new Date();
                const riskLevel = data.alert_level; // Get actual risk level from backend
                const riskText = document.querySelector(".risk-text"); // Element to update risk text
                const riskCircle = document.querySelector(".live-notify");

                // Update the chart with the new data
                riskChart.data.labels.push(now);
                riskChart.data.datasets[0].data.push(riskLevelNumeric(riskLevel));  // Convert string to numeric value

                if (riskChart.data.labels.length > 10) {
                    riskChart.data.labels.shift();
                    riskChart.data.datasets[0].data.shift();
                }

                riskChart.update();

                // 🔹 Update risk notification text and color based on the string level
                if (riskLevel === "Low") {
                    riskText.textContent = "Low Risk... Safe Flight";
                    riskCircle.style.backgroundColor = "#6d9e3b"; // Green for low risk
                } else if (riskLevel === "Medium") {
                    riskText.textContent = "Moderate Risk... Caution";
                    riskCircle.style.backgroundColor = "#f39c12"; // Yellow for moderate risk
                } else if (riskLevel === "High") {
                    riskText.textContent = "High Risk... Not Safe";
                    riskCircle.style.backgroundColor = "#e74c3c"; // Red for high risk
                }
            })
            .catch(error => console.error("Error fetching risk data:", error));
    }

    function riskLevelNumeric(riskLevel) {
        if (riskLevel === "Low") {
            return 3;  // Example numeric value for low risk
        } else if (riskLevel === "Medium") {
            return 6;  // Example numeric value for medium risk
        } else if (riskLevel === "High") {
            return 9;  // Example numeric value for high risk
        } else {
            return 0;  // Default in case of an unexpected value
        }
    }

    setInterval(updateChart, 5000);
});
