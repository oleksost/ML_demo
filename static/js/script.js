var valueSelected = "Person1";
//var viz_basic = VizBasic('container_iframe');

$('select').change(function () {
     var optionSelected = $(this).find("option:selected");
     valueSelected  = String(optionSelected.val());
     //var textSelected   = optionSelected.text();
    console.log(valueSelected);
 });


$(window).on("load", function() {
    console.log("window loaded");
});

$( "#train_initial" ).click(function() {
    $.ajax({
        type: 'POST',
        url: "/train_model",
        dataType: 'json',
        data: {
            person: valueSelected
        },
        }).done(function(data) {
        console.log("done");
        var data_j = JSON.parse(data);
        mpld3.draw_figure('container_iframe', data_j);
        console.log(data_j.data);

    })
    });

$( "#add_point_to_model" ).click(function() {
    $.ajax({
        type: 'POST',
        url: "/add_point",
        dataType: 'json',
        data: {
            person: valueSelected
        },
        }).done(function(data) {
        console.log("done");
        var data_j = JSON.parse(data);
        mpld3.draw_figure('container_iframe', data_j);
        console.log(data_j.data);

    })
    });
$( "#new_dat_point" ).click(function() {
    $.ajax({
        type: 'POST',
        url: "/create_new_datapoint",
        data: {
            person: valueSelected
        },
        }).done(function(data) {
        var data_j = JSON.parse(data);
        /*mpld3.draw_figure('container_iframe', data_j);*/
        $( "#long" ).html( data_j.longitude );
        $( "#lat" ).html( data_j.latitude );
        $( "#dist" ).html( data_j.distance_home);
        $( "#weekday" ).html( data_j.weekday );
        $( "#hour" ).html( data_j.hour );
        $( "#min" ).html( data_j.minutes );
        $( "#temp" ).html( data_j.apparentTemperature);
        $( "#time" ).html( data_j.time);
        $( "#downfall" ).html( data_j.downfall);
        console.log(data_j);

    })
    });

//STOP EXECUTION ON RELOAD
window.onbeforeunload = function(event) {
    stop_demo();
};


