var serialJSON = { made: 1 };
var mode;
console.log("hhii");

function successful(mode) {
  serialJSON["mode"] = mode;

  $.ajax({
    url: "https://natours-production-f625.up.railway.app/api/v1/tours/coffeeTest/76231",
    type: "GET",

    success: function (res) {
      console.log(res);
    },

    error: function (res) {
      alert("There has been a error! please refresh the page and try again.");
    },
  });
}

successful(1);
