(async () => {
  waitTillAppears(".guess-map__guess-button");
  await wait(1000);
  hideGUI(true);

  chrome.runtime.sendMessage(
    {
      action: "screenshot",
    },
    (response) => {
      log(response.image);
      debugBase64(response.image);
      hideGUI(false);
    }
  );
  waitTillDisappears(".guess-map__guess-button");
})();

async function wait(millis) {
  await new Promise((r) => setTimeout(r, millis));
}

async function waitTillAppears(selector) {
  while (!document.querySelector(selector)) {
    (async () => {
      await new Promise((r) => setTimeout(r, 100));
    })();
  }
}

async function waitTillDisappears(selector) {
  while (!!document.querySelector(selector)) {
    (async () => {
      await new Promise((r) => setTimeout(r, 200));
    })();
  }
}

function log(content) {
  chrome.runtime.sendMessage(
    { action: "log", content: content },
    function (response) {}
  );
}

function hideGUI(hide) {
  const view = hide ? "none" : "";
  var others = document.querySelectorAll(
    ".game-layout__guess-map,.gm-control-active,.gm-compass,.gmnoprint,.gm-style-cc,.gm-bundled-control, .game-layout__controls, .game-layout__status, .guess-map__toggle, .game-layout__top-hud"
  );
  var print = document.getElementsByClassName("gmnoprint");
  for (var i = 0; i < others.length; i++) {
    try {
      others[i].style.display = view;
    } catch (e) {}
  }
  for (var i = 0; i < print.length; i++) {
    try {
      print[i].children[0].style.display = view;
    } catch (e) {}
  }
}

function debugBase64(base64URL) {
  var win = window.open();
  win.document.write(
    '<iframe src="' +
      base64URL +
      '" frameborder="0" style="border:0; top:0px; left:0px; bottom:0px; right:0px; width:100%; height:100%;" allowfullscreen></iframe>'
  );
}
