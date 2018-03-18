$(document).ready(function(){
	$("#use_faceswap").click(function() {
		chrome.tabs.executeScript(null, {file: 'script_for_page.js'});
	})
 });
