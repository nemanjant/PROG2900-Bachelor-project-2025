const express = require("express");
const fs = require("fs");
const path = require("path");

const app = express();
const port = 3000;

// Create a base folder to store session data
const baseDataFolder = path.join(__dirname, "questionnaire_sessions");
if (!fs.existsSync(baseDataFolder)) {
  fs.mkdirSync(baseDataFolder);
}

app.use(express.json()); // Parse JSON request bodies

app.use(express.static("public")); // Serve static files from the "public" folder

// Endpoint to save data
app.post("/save-data", (req, res) => {
  const { sessionId, data } = req.body;

  // Log the received data for debugging
  console.log("Received data:", data);

  // Validate required fields
  if (!sessionId || !data || !data.question || !data.answer || !data.mouseMovements) {
    return res.status(400).json({ message: "Missing required fields in request body" });
  }

  const sessionFolder = path.join(baseDataFolder, `session_${sessionId}`);

  // Create a new folder for the session if it doesn't exist
  if (!fs.existsSync(sessionFolder)) {
    fs.mkdirSync(sessionFolder);
  }

  // Create separate folders for truthful and untruthful parts
  const partFolder = path.join(sessionFolder, `part_${data.part || 1}`); // Default to part_1 if not provided
  if (!fs.existsSync(partFolder)) {
    fs.mkdirSync(partFolder);
  }

  // Generate a filename based on the question
  const fileName = `question_${data.question.replace(/[^a-zA-Z0-9]/g, "_")}.json`;
  const filePath = path.join(partFolder, fileName);

  // Prepare the data to be saved
  const saveData = {
    question: data.question,
    answer: data.answer,
    mouseMovements: data.mouseMovements || [], // Ensure mouseMovements is included
    timestamps: data.timestamps || [], // Ensure timestamps is included
    accelerations: data.accelerations || [], // Ensure accelerations is included
    jerks: data.jerks || [], // Ensure jerks is included
    curvatures: data.curvatures || [], // Ensure curvatures is included
    pausePoints: data.pausePoints || [], // Ensure pausePoints is included
    hesitation: data.hesitation || 0, // Ensure hesitation is included
    hesitationLevel: data.hesitationLevel,
    totalTime: data.totalTime || 0, // Ensure totalTime is included
    averageSpeed: data.averageSpeed || 0, // Ensure averageSpeed is included
    deceptionFlag: data.deceptionFlag, // Response may be suspicious and require further review
    jerkSpikeCount: data.jerkSpikeCount || 0 
  };

  // Save the data to a JSON file
  fs.writeFile(filePath, JSON.stringify(saveData, null, 2), (err) => {
    if (err) {
      console.error("Error saving file:", err);
      return res.status(500).json({ message: "Failed to save data" });
    }
    console.log(`Data saved to: ${filePath}`);
    res.json({ message: "Data saved successfully", file: fileName });
  });
});

// Start the server
app.listen(port, () => {
  console.log(`Server is running on http://localhost:${port}`);
});