function showContent(contentType) {
    // Hide both sections initially
    document.getElementById('report_content').style.display = 'none';
    document.getElementById('schedule_content').style.display = 'none';

    // Remove active class from both tabs
    document.getElementById('new_report_btn').classList.remove('active');
    document.getElementById('schedule_btn').classList.remove('active');

    // Show the selected tab's content and set the active class on the selected tab
    if (contentType === 'report') {
        document.getElementById('report_content').style.display = 'block';
        document.getElementById('new_report_btn').classList.add('active');
    } else if (contentType === 'schedule') {
        document.getElementById('schedule_content').style.display = 'block';
        document.getElementById('schedule_btn').classList.add('active');
    }
}

document.getElementById("scheduleButton").addEventListener("click", function() {
    // Get form values
    var reportTitle = document.getElementById("reportTitle").value;
    var reportType = document.getElementById("reportType").value;
    var fromDate = document.getElementById("fromDate").value;
    var toDate = document.getElementById("toDate").value;
    var scheduleDate = document.getElementById("scheduleDate").value;

    // Get filter values
    var includeSize = document.getElementById("includeSize").checked ? "Bird Size Included" : "No Bird Size";
    var includeBirdCount = document.getElementById("includeBirdCount").checked ? "Bird Count Included" : "No Bird Count";
    var includeWeather = document.getElementById("includeWeather").checked ? "Weather Conditions Included" : "No Weather Conditions";
    var riskLevel = document.getElementById("riskLevel").value;

    // Create the content string for the table
    var content = `${includeSize}, ${includeBirdCount}, ${includeWeather}, Risk Level: ${riskLevel}`;

    // Create a new row in the table
    var table = document.getElementById("scheduleHistoryTable");
    var newRow = table.insertRow();

    // Insert cells with data
    newRow.insertCell(0).textContent = scheduleDate;
    newRow.insertCell(1).textContent = reportTitle;
    newRow.insertCell(2).textContent = `Type: ${reportType}, Date Range: ${fromDate} to ${toDate}, ${content}`;

    // Reset the form and close the modal
    document.getElementById("scheduleForm").reset();
    $('#scheduleModal').modal('hide');
    
    // Show the schedule content
    document.getElementById("schedule_content").style.display = "block";
});

