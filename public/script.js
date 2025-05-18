const questionsPart1 = [
  "Are you currently located in Norway?",
  "Are you currently located in the county Innlandet?",
  "Are you currently located in Gjovik?",
  "Are you currently located at NTNU?",
  "Are you currently a student at NTNU?",
  "Are you currently located in the USA?",
  "Are you currently located in the state California?",
  "Are you currently located in Berkeley?",
  "Are you currently located at UC, Berkeley?",
  "Are you currently a student at UC, Berkeley?",
];

const questionsPart2 = [
  "Are you currently located in Australia?",
  "Are you currently located in the state Victoria?",
  "Are you currently located in Melbourne?",
  "Are you currently located at RMIT?",
  "Are you currently employed as a professor at RMIT?",
  "Are you currently located in Norway?",
  "Are you currently located in the county Innlandet?",
  "Are you currently located in Gjovik?",
  "Are you currently located at NTNU?",
  "Are you currently a student at NTNU?",
];

let currentPart = 1;
let currentQuestionIndex = 0;
// Stores points where the mouse current is, x y coordinates
let mouseData = [];
let startTime = null;
let lastPosition = { x: 0, y: 0 };
let lastTime = null;
let isTracking = false;
let sessionId = null;

// Stores points where the mouse pauses
let pausePoints = [];

// Total hesitation time in seconds
let hesitation = 0; 

// Timestamp of the last mouse movement
let lastMoveTime = null; 

// Define max jerk threshold
const MAX_JERK = 1000000;

// Adjust the number of previous points used for smoothing
const SMOOTHING_WINDOW = 3;

const startBtn = document.getElementById("start-btn"); 
const instructionScreen = document.getElementById("instruction-screen");
const instructionText = document.getElementById("instruction-text");
const continueBtn = document.getElementById("continue-btn");
const thankYouScreen = document.getElementById("thank-you-screen");
const proceedBtn = document.getElementById("proceed-btn");
const questionScreen = document.getElementById("question-screen");
const questionText = document.getElementById("question-text");
const yesBtn = document.getElementById("yes-btn");
const noBtn = document.getElementById("no-btn");
const nextBtn = document.getElementById("next-btn");
const resultScreen = document.getElementById("result-screen");

startBtn.addEventListener("click", startQuestionnaire);
continueBtn.addEventListener("click", startPart);
proceedBtn.addEventListener("click", startSecondPart);
yesBtn.addEventListener("click", () => handleAnswer("Yes"));
noBtn.addEventListener("click", () => handleAnswer("No"));
nextBtn.addEventListener("click", showNextQuestion);

// Starting questionare
function startQuestionnaire() {
  document.getElementById("start-screen").style.display = "none";
  instructionScreen.style.display = "block";
  instructionText.innerHTML  = "We ask that you answer these next ten questions truthfully";
  sessionId = Date.now();
}

// Starting questionare, truth part
function startPart() {
  instructionScreen.style.display = "none";
  questionScreen.style.display = "block";
  showQuestion();
  startTracking();
}

// Starting questionare, lie part
function startSecondPart() {
  thankYouScreen.style.display = "none";
  instructionScreen.style.display = "block";
  instructionText.textContent = "For the next ten questions we ask that you answer the questions untruthfully (LIE)";
}

function showQuestion() {
  const questions = currentPart === 1 ? questionsPart1 : questionsPart2;
  questionText.textContent = questions[currentQuestionIndex];
  nextBtn.style.display = "none";
}

function showNextQuestion() {
  currentQuestionIndex++;
  const questions = currentPart === 1 ? questionsPart1 : questionsPart2;
  if (currentQuestionIndex < questions.length) {
    showQuestion();
    startTracking();
  } else {
    if (currentPart === 1) {
      currentPart = 2;
      currentQuestionIndex = 0;
      questionScreen.style.display = "none";
      thankYouScreen.style.display = "block";
    } else {
      questionScreen.style.display = "none";
      resultScreen.style.display = "block";
    }
  }
}

function handleAnswer(answer) {
  stopTracking();
  const endTime = Date.now();
  const totalTime = (endTime - startTime) / 1000;
  const speed = calculateSpeed(mouseData);

  // Calculate jerk spike threshold dynamically
  const jerkValues = mouseData.map(p => Math.abs(p.jerk));
  const meanJerk = jerkValues.reduce((a, b) => a + b, 0) / (jerkValues.length || 1);
  const dynamicJerkThreshold = Math.min(Math.max(meanJerk * 4, 400000), 1000000);

  //Count spikes above the threshold
  const highJerkCount = jerkValues.reduce((count, jerk) => {
    return jerk > dynamicJerkThreshold ? count + 1 : count;
  }, 0);

  // Combined logic to determine hesitation level
  let hesitationLevel = "low";
  if (hesitation > 3 || pausePoints.length > 5 || highJerkCount > 16) {
    hesitationLevel = "high";
  } else if (hesitation > 2 || (highJerkCount >= 11 && highJerkCount <= 16)) {
    hesitationLevel = "moderate";
  }

  // Flag for possible deception
  const possibleDeception = hesitationLevel === "high";

  // Log for debugging
  console.log(`Hesitation Time: ${hesitation.toFixed(3)} sec`);
  console.log(`Pause Count: ${pausePoints.length}`);
  console.log(`Jerk Spikes: ${highJerkCount}`);
  console.log(`erk Threshold: ${dynamicJerkThreshold.toFixed(2)}`);
  console.log(`Hesitation Level: ${hesitationLevel}`);
  console.log(`Deception Indicator: ${possibleDeception ? "High" : "Low"}`);

  const sessionData = {
    part: currentPart,
    question: (currentPart === 1 ? questionsPart1 : questionsPart2)[currentQuestionIndex],
    answer: answer,
    mouseMovements: mouseData.map(point => [point.x, point.y]),
    timestamps: mouseData.map(point => point.time),
    accelerations: mouseData.map(point => point.acceleration),
    jerks: mouseData.map(point => point.jerk),
    curvatures: mouseData.map(point => point.curvature),
    pausePoints: pausePoints,
    hesitation: hesitation,
    hesitationLevel: hesitationLevel,
    totalTime: totalTime,
    averageSpeed: speed,
    deceptionFlag: possibleDeception,
    jerkSpikeCount: highJerkCount,                  
    jerkSpikeThreshold: dynamicJerkThreshold        
  };

  saveDataToFile(sessionData);
  nextBtn.style.display = "block";
}

function startTracking() {
  startTime = Date.now();
  lastTime = startTime;
  lastPosition = { x: 0, y: 0 };
  mouseData = [];
  pausePoints = [];
  hesitation = 0;
  lastMoveTime = null;
  isTracking = true;
  document.addEventListener("mousemove", trackMouse);
}

function stopTracking() {
  isTracking = false;
  document.removeEventListener("mousemove", trackMouse);
}

function trackMouse(event) {
  if (!isTracking) return;

  const currentTime = Date.now();
  const timeDiff = (currentTime - lastTime) / 1000; // Convert to seconds
  const currentPosition = { x: event.clientX, y: event.clientY };

  if (timeDiff > 0) {
    const distance = Math.sqrt(
      Math.pow(currentPosition.x - lastPosition.x, 2) + Math.pow(currentPosition.y - lastPosition.y, 2)
    );

    if (distance < 2) { // Mouse barely moved, possible hesitation
      if (lastMoveTime === null) {
        lastMoveTime = currentTime; // Start of a pause
      } else {
        const pauseDuration = (currentTime - lastMoveTime) / 1000; // Convert ms to sec
        
        // Apply research-based thresholds
        if (pauseDuration > 0.2) {  
          pausePoints.push({ x: currentPosition.x, y: currentPosition.y, duration: pauseDuration });
          hesitation += pauseDuration;

          console.log(`Pause at (${currentPosition.x}, ${currentPosition.y}) for ${pauseDuration.toFixed(3)} sec`);
        }
      }
    } else {
      lastMoveTime = null; // Reset pause tracking if mouse moves
    }

    const speed = calculateSpeedBetweenPoints(lastPosition, currentPosition, timeDiff);
    const acceleration = calculateAcceleration(lastPosition, currentPosition, timeDiff, speed);
    const jerk = calculateJerk(acceleration, timeDiff);
    const curvature = calculateCurvature(lastPosition, currentPosition, timeDiff);

    mouseData.push({ x: currentPosition.x, y: currentPosition.y, time: (currentTime - startTime) / 1000, speed, acceleration, jerk, curvature });

    lastPosition = currentPosition;
    lastTime = currentTime;
  }
}

function calculateSpeedBetweenPoints(startPos, endPos, time) {
  const distance = Math.sqrt(
    Math.pow(endPos.x - startPos.x, 2) + Math.pow(endPos.y - startPos.y, 2)
  );
  return distance / time; // Speed in pixels per second
}

function calculateAcceleration(startPos, endPos, time, currentSpeed) {
  if (mouseData.length === 0) return 0;

  const lastSpeed = mouseData[mouseData.length - 1].speed;
  const acceleration = (currentSpeed - lastSpeed) / time;

  return acceleration;
}

function calculateJerk(currentAcceleration, timeDiff) {
  // Early exit check, prevents errors
  if (mouseData.length === 0 || timeDiff < 0.01) return 0;

  // Get recent accelerations for smoothing
  const recentAccelerations = mouseData
    .slice(Math.max(0, mouseData.length - SMOOTHING_WINDOW))
    .map(p => p.acceleration);

  // Include the current acceleration as well
  recentAccelerations.push(currentAcceleration);

  // Averages all recent values to avoid random spikes or noise.
  const smoothedAcceleration =
    recentAccelerations.reduce((sum, a) => sum + a, 0) / recentAccelerations.length;
  
  // Difference in acceleration, divided by time
  const lastAcceleration = mouseData[mouseData.length - 1].acceleration || 0;
  let jerk = (smoothedAcceleration - lastAcceleration) / timeDiff;

  // Clamp extreme jerk values
  jerk = Math.max(Math.min(jerk, MAX_JERK), -MAX_JERK);

  return jerk;
}

function calculateCurvature(startPos, endPos, timeDiff) {
  // Need at least 3 points
  // Three points allows to measure how the direction bends, 2 points straight line, third gives bend
  if (mouseData.length < 3) return 0;

  // Second last point
  const prevPoint = mouseData[mouseData.length - 2];

  // Last point
  const lastPoint = mouseData[mouseData.length - 1]; 

  // First derivatives, velocity (x', y')
  const dx1 = lastPoint.x - prevPoint.x;
  const dy1 = lastPoint.y - prevPoint.y;
  const dx2 = endPos.x - lastPoint.x;
  const dy2 = endPos.y - lastPoint.y;

  // Second derivatives, acceleration (x'', y'')
  const ddx = dx2 - dx1;
  const ddy = dy2 - dy1;

  const numerator = Math.abs(dx1 * ddy - dy1 * ddx);
  const denominator = Math.pow(dx1 * dx1 + dy1 * dy1, 1.5);

  // 1e-6 guard avoids division by near-zero values, or 0
  if (denominator < 1e-6) return 0;

  const curvature = numerator / denominator;
  
  return curvature;
}

function calculateSpeed(mouseData) {
  // If there’s only one point or none, there’s no movement, speed=0
  if (mouseData.length < 2) return 0;

  // Initializes accumulators for distance and time
  let totalDistance = 0;
  let totalTime = 0;

  // Iterates over all pairs of consecutive points
  for (let i = 1; i < mouseData.length; i++) {
    const prev = mouseData[i - 1];
    const curr = mouseData[i];

    // Euclidean distance between prev and curr point
    const distance = Math.sqrt(
      Math.pow(curr.x - prev.x, 2) + Math.pow(curr.y - prev.y, 2)
    );
    totalDistance += distance;

    // Adds time difference in seconds
    totalTime += (curr.time - prev.time);
  }

  // Average speed in pixels per second
  return totalDistance / totalTime;
}

// Saving data as json file
function saveDataToFile(data) {
  fetch("/save-data", {
    method: "POST",
    headers: {
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ sessionId, data }),
  })
    .then((response) => response.json())
    .then((result) => {
      console.log("Data saved successfully:", result);
    })
    .catch((error) => {
      console.error("Error saving data:", error);
    });
}