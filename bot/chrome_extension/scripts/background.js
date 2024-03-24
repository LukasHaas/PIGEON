//Windows resize
chrome.runtime.onMessage.addListener(function (request, sender, sendResponse) {
  if (request && request.action === "resizeWindow") {
    chrome.windows.getCurrent(function (window) {
      var updateInfo = {
        width: 921,
        height: 1000,
      };
      (updateInfo.state = "normal"),
        chrome.windows.update(window.id, updateInfo);
    });
  } else if (request && request.action === "screenshot") {
    chrome.tabs.captureVisibleTab(null, {}, function (dataUri) {
      sendResponse({ image: dataUri });
    });
    return true;
  } else if (request && request.action === "log") {
    console.log(request.content);
  }
});