(async () => {
  chrome.runtime.sendMessage(
    {
      action: "resizeWindow",
    },
    function (createdWindow) {
      console.log("Window Resize");
    }
  );

  let currentRoundNumber = 1;

  while (true) {
    await waitTillAppears(".guess-map__guess-button");
    await wait(randomIntFromInterval(2000, 9000));

    hideGUI(true);
    await changeHeading(0);
    await wait(1250);
    const response1 = await screenshot();
    const image1 = response1.image;
    hideGUI(false);
    await wait(250);

    hideGUI(true);
    await changeHeading(90);
    await wait(1250);
    const response2 = await screenshot();
    const image2 = response2.image;
    hideGUI(false);
    await wait(250);

    hideGUI(true);
    await changeHeading(180);
    await wait(1250);
    const response3 = await screenshot();
    const image3 = response3.image;
    hideGUI(false);
    await wait(250);

    hideGUI(true);
    await changeHeading(270);
    await wait(1250);
    const response4 = await screenshot();
    const image4 = response4.image;
    hideGUI(false);
    await wait(250);

    // await wait(2000);

    const apiResp = await fetch("http://127.0.0.1:5000/api/v1/predict", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        gameID: getGameID(),
        roundID: currentRoundNumber,
        image: image1,
        image_2: image2,
        image_3: image3,
        image_4: image4,
      }),
    });
    const guess = await apiResp.json();
    console.log(guess);

    let result;
    do {
      result = await submitGuess(guess.results.lat, guess.results.lng, currentRoundNumber);
      console.log(result.body);
      if (!result.body.currentRoundNumber) {
        currentRoundNumber += 1;
      } else {
        currentRoundNumber = result.body.currentRoundNumber + 1;
      }

    } while (
      result.resp.status == 400
    );

    var guessButton = ".guess-map__guess-button";
    if (window.location.href.includes('battle-royale')) {
      guessButton = "[class^=game_guess]";
    }

    await fetch("http://127.0.0.1:5000/api/v1/game", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify({
        gameID: getGameID(),
        roundID: currentRoundNumber - 1,
        game: result.body
      }),
    });

    await waitTillDisappears(guessButton);
  }
})();

function screenshot() {
  return new Promise((resolve) => {
    chrome.runtime.sendMessage(
      {
        action: "screenshot",
      },
      (response) => resolve(response)
    );
  });
}

function getGameID() {
  const urlSplit = window.location.href.split("/");
  const gameID = urlSplit[urlSplit.length - 1];
  return gameID;
}

function randomIntFromInterval(min, max) { // min and max included 
  return Math.floor(Math.random() * (max - min + 1) + min)
}

async function wait(millis) {
  await new Promise((r) => setTimeout(r, millis));
}

async function waitTillAppears(selector) {
  while (!document.querySelector(selector)) {
    await wait(100);
  }
}

async function waitTillDisappears(selector) {
  while (document.querySelector(selector)) {
    await wait(100);
  }
}

async function submitGuess(lat, lng, roundNumber) {
  const gameID = getGameID();
  apiURL = "https://game-server.geoguessr.com/api/duels/" + gameID + "/guess";
  if (window.location.href.includes('battle-royale')) {
    apiURL = "https://game-server.geoguessr.com/api/battle-royale/" + gameID + "/guess";
  }

  const payload = {
    lat: lat,
    lng: lng,
    roundNumber: roundNumber,
  };

  const headers = {
    origin: "https://www.geoguessr.com",
    referer: apiURL,
    "accept-language": "en-GB,en-US;q=0.9,en;q=0.8",
    "sec-ch-ua":
      '"Chromium";v="106", "Google Chrome";v="106", "Not;A=Brand";v="99"',
    "sec-ch-ua-mobile": "?0",
    "sec-ch-ua-platform": "macOS",
    "user-agent":
      "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/106.0.0.0 Safari/537.36",
    "x-client": "web",
    "Content-Type": "application/json",
  };

  const resp = await fetch(apiURL, {
    method: "POST",
    credentials: "include",
    headers: headers,
    body: JSON.stringify(payload),
  });
  const body = await resp.json();
  return { resp, body };
}

function simulateMouseEvent(element, eventName, X, Y) {
  element.dispatchEvent(
    new MouseEvent(eventName, {
      view: window,
      bubbles: true,
      ancelable: true,
      clientX: X,
      clientY: Y,
      button: 0,
    })
  );
}
async function changeHeading(degrees) {
  await waitTillAppears('[data-qa="compass"]');
  let compass = document.querySelector('[data-qa="compass"]');

  let box = compass.getBoundingClientRect();
  let X = box.left + (box.right - box.left) / 2;
  let Y = box.top + (box.bottom - box.top) / 2;

  let angle = ((degrees - 90) / 180) * Math.PI;
  X += 1000 * Math.cos(angle);
  Y += 1000 * Math.sin(angle);

  simulateMouseEvent(compass, "mousedown", X, Y);
  simulateMouseEvent(compass, "mouseup", X, Y);
  simulateMouseEvent(compass, "click", X, Y);
}

function log(content) {
  chrome.runtime.sendMessage(
    { action: "log", content: content },
    function (response) {}
  );
}

function hideGUI(hide) {
  const view = hide ? "none" : "";
  let paths = document.getElementsByTagName("path");
  for (let i = 0; i < paths.length; i++) {
    paths[i].style.display = view;
  }

  let mentions = document.getElementsByClassName("gmnoprint");
  for (let i = 0; i < mentions.length; i++) {
    mentions[i].style.display = view;
  }

  let controls = document.querySelectorAll("[class^=game-panorama_controls]");
  for (let i = 0; i < controls.length; i++) {
    controls[i].style.display = view;
  }

  let o_controls = document.querySelectorAll("[class^=game_controls]");
  for (let i = 0; i < o_controls.length; i++) {
    o_controls[i].style.display = view;
  }

  let o_map = document.querySelectorAll("[class^=game_guess]");
  for (let i = 0; i < o_map.length; i++) {
    o_map[i].style.display = view;
  }

  let guessMap = document.querySelectorAll("[class^=game-map]");
  for (let i = 0; i < guessMap.length; i++) {
    guessMap[i].style.display = view;
  }

  let chat = document.querySelectorAll("[class^=chat-input]");
  for (let i = 0; i < chat.length; i++) {
    chat[i].style.display = view;
  }

  let msg = document.querySelectorAll("[class^=chat-message]");
  for (let i = 0; i < msg.length; i++) {
    msg[i].style.display = view;
  }

  let hud = document.querySelectorAll("[class^=game_hud]");
  for (let i = 0; i < hud.length; i++) {
    hud[i].style.display = view;
  }

  let consent = document.getElementById("adconsent-usp-link");
  if (consent) {
    consent.style.display = view;
  }
}

function debugBase64(base64URL) {
  let win = window.open();
  win.document.write(
    '<iframe src="' +
      base64URL +
      '" frameborder="0" style="border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;" allowfullscreen></iframe>'
  );
}
