const express = require("express");
const bodyParser = require("body-parser");
const { spawn } = require("child_process");

const app = express();
app.use(bodyParser.json()); // Parsing request body sebagai JSON

// Endpoint untuk prediksi
app.post("/predict", (req, res) => {
    const userInput = req.body;

    // Validasi input JSON
    if (!userInput || !userInput.location || userInput.location.length !== 2) {
        return res.status(400).json({
            error: "Invalid input. JSON must include a 'location' array with [latitude, longitude].",
        });
    }

    const pythonProcess = spawn("python", ["model/model.py"]);

    let pythonOutput = "";
    let pythonError = "";

    // Kirim input JSON ke script Python melalui stdin
    pythonProcess.stdin.write(JSON.stringify(userInput));
    pythonProcess.stdin.end();

    // Tangkap output dari Python
    pythonProcess.stdout.on("data", (data) => {
        pythonOutput += data.toString();
    });

    // Tangkap error dari Python
    pythonProcess.stderr.on("data", (data) => {
        pythonError += data.toString();
        console.error("Python error:", pythonError);
    });

    // Tangani proses setelah script Python selesai
    pythonProcess.on("close", (code) => {
        if (code !== 0) {
            return res.status(500).json({
                error: "Internal server error. Please check the Python script.",
                details: pythonError.trim(),
            });
        }

        try {
            const predictions = JSON.parse(pythonOutput); // Parse JSON dari output Python
            res.json(predictions); // Kirim respons JSON ke Postman
        } catch (err) {
            res.status(500).json({
                error: "Failed to parse prediction output.",
                details: err.message,
            });
        }
    });
});

// Jalankan server pada port 3000
const PORT = 3000;
app.listen(PORT, () => {
    console.log(`Server running on http://localhost:${PORT}`);
});
