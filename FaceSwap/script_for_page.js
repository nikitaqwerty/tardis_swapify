$("body").prepend("<div class='popup'>" +
                    "<div class='popup__box'>" +
                        "<div id='cd-login'>" +
                            "<form id='login_form' class='cd-form' action='photo' method='post' >" +
                                "<p class='fieldset'>" +
                                    "<label for='file-upload' class='custom-file-upload'>" +
                                        "<i class='fa fa-cloud-upload'></i> Загрузить фотографию" +
                                    "</label>" +
                                    "<p class='fieldset'>" +
                                        "<img id='user-photo' style='display: none;'/>" +
                                    "</p>" +
                                    "<input id='file-upload' name='photo' type='file' style='display:none;'>" +
                                "</p>" +
                                "<p class='fieldset'>" +
                                    "<input class='button-faceswap' id='load' style='display: none;' type='button' value='Загрузить'/>" +
                                "</p>" +
                            "</form>" +
                        "</div>" +
                    "</div>" +
                "</div>")
$(".popup").css({"opacity": 1, "visibility": "visible"});
$(".popup .popup__box").css({"-webkit-transform": "rotate(0deg) translate(0, 0)", "transform": "rotate(0deg) translate(0, 0)"});


function readURL(input) {
  if (input.files && input.files[0]) {
    var reader = new FileReader();
    reader.onload = function(e) {
      $('#user-photo').attr('src', e.target.result);
    }
    reader.readAsDataURL(input.files[0]);
  }
}
URL = "https://localhost:4443/"
$('#file-upload').change(function() {
    readURL(this);
	var i = $(this).prev('label').clone();
	var file = $('#file-upload')[0].files[0].name;
	$(this).prev('label').text(file);
	$("#load").removeAttr("style");
    $("#user-photo").removeAttr("style");
    $("#user-photo").css({"height": "10em"});
});

$("#load").on('click', function () {
    var fd = new FormData();
    fd.append('photo', $('#file-upload').prop('files')[0]);
    fd.append('link', $("img#galleryImage-0").attr("src").toString());

    $(".popup").css({"opacity": 1, "visibility": "visible", "background": "rgba(255,255,255,0.6)"});
    $('.popup').html("<img src='https://meas.tardis3d.ru/images/loader.gif'/>")
    $.ajax({
        processData: false,
        contentType: false,
        type: "POST",
        url: URL + "photo",
        data : fd,
        success: function(photo) {
            $("img#galleryImage-0").attr("src",URL + photo)
            $(".popup").remove();
        }
    });
})

$('#load').on('change invalid', function() {
    return false
});