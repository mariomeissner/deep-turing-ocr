// Global variables
var total_proportion;
var file;
var boxes = [];
var lines = [];
var ids = [];
var topOffset, leftOffset;
var stage = 0;

// PARAMETERS
MIN_WIDTH = 80;

$(document).ready(function () {
  $("#loader").css("visibility", "hidden");
  $("#crop").css("visibility", "hidden");
  var rect = document.getElementById('image-col').getBoundingClientRect();
  topOffset = rect.top;
  leftOffset = rect.left;
})

var URL = "http://localhost:5000/"

function loadAnimation() {
  $("#loader").css("visibility", "visible");
}

function loadImage() {
  clearAll()
  stage = 1;
  $("#crop").css("visibility", "visible");
  total_proportion = 1;
  file = document.querySelector('input[type=file]').files[0];

  reader = new FileReader();
  reader.onloadend = function () {

    img = document.getElementById('crop');
    img.src = reader.result;

    img.onload = function () {
      height = img.naturalHeight;
      width = img.naturalWidth;

      // Make sure size is smaller than document height
      if (height > window.innerHeight - 100) {
        old_height = height;
        height = window.innerHeight - 100;
        proportion = height / old_height;
        width = width * proportion;
        total_proportion = proportion;
      }

      // Make sure size is smaller than 60% width
      max_width = Math.round(window.innerWidth * 0.7)
      if (width > max_width) {
        old_width = width;
        width = max_width;
        proportion = width / old_width;
        height = height * proportion;
        total_proportion = total_proportion * proportion;
      }

      // Set the size parameters
      img.style.width = width + 'px';
      img.style.height = height + 'px';

      $("#loader").css("visibility", "hidden");
    }
  }

  if (file) {
    reader.readAsDataURL(file);
  } else {
    alert("No File found :(")
  }
}

function printBoxes() {

  clearAll();
  if (stage < 1) {
    alert("Please load an image first!");
    return;
  }

  stage = 2;

  url = URL + "predict_boxes?file=" + file.name
  boxes = [];

  $("#loader").css("visibility", "visible");
  fetch(url)
    .then(function (response) {
      return response.json();
    })
    .then(function (coords) {
      coords = purge(coords);
      coords = clusterSortPoints(coords);

      // Display boxes on the image
      for (i = 0; i < coords.length; i++) {
        var coord = coords[i]

        // Calculate coordinates
        div = document.createElement("div");
        header = document.createElement("div");
        div.append(header)
        left = coord[0] * total_proportion;
        _top = coord[1] * total_proportion;
        width = coord[2] * total_proportion - left;
        height = coord[3] * total_proportion - _top;

        // Apply offsets
        left += leftOffset;
        _top += topOffset;
        width += leftOffset;
        height += topOffset;

        // Create the div
        div.className = 'box';
        header.className = 'header';
        dragElement(div)
        div.style.top = _top + "px";
        div.style.left = left + "px";
        div.style.width = width + "px";
        div.style.height = height + "px";
        div.id = 'box-' + i;

        // To remove the box, doubleclick the header
        $(header).dblclick(function () {
          $(this).parent().remove()
        })

        // Highlight line associated to box
        $(div).hover(
          function () {
            box = $(this)
            id = box.attr('id')
            id_num = id.split("-")[1]
            line = $("#line-" + id_num)
            line.addClass('hover')
            box.addClass('hover')
          },
          function () {
            box = $(this)
            id = box.attr('id')
            id_num = id.split("-")[1]
            line = $("#line-" + id_num)
            line.removeClass('hover')
            box.removeClass('hover')
          }
        )

        imagecol = document.getElementById("image-col")
        imagecol.appendChild(div);
        boxes.push(div)
      }
      $("#loader").css("visibility", "hidden");
    })
}

function purge(points) {
  var purged = [];
  for (point of points) {
    if (point[2] - point[0] > MIN_WIDTH) {
      purged.push(point);
    }
  }
  return purged;
}

function predictLines() {

  if (stage < 2) {
    alert("Please get and adjust the boxes first!");
    return;
  }

  stage = 3;

  clearLines();
  $("#loader").css("visibility", "visible");
  tuples = [];
  ids = [];
  for (box of boxes) {
    if (!$.contains(document, box)) { continue; }
    var _top = -topOffset + Math.round(box.style.top.split("px")[0]);
    var left = -leftOffset + Math.round(box.style.left.split("px")[0]);
    var bottom = -topOffset + Math.round(box.style.height.split("px")[0]);
    var right = -leftOffset + Math.round(box.style.width.split("px")[0]);
    _top = Math.round(_top / total_proportion);
    left = Math.round(left / total_proportion);
    bottom = Math.round(bottom / total_proportion + _top);
    right = Math.round(right / total_proportion + left);
    tuple = [left, _top, right, bottom];
    tuples.push(tuple);
    ids.push($(box).attr('id').split('-')[1])
  }

  $.ajax({
    type: "POST",
    url: URL + "predict_lines?file=" + file.name,
    data: JSON.stringify(tuples),
    contentType: "application/json; charset=utf-8",
    dataType: "json",
    success: function (data) {
      predictLinesResult(data)
    },
    failure: function (errMsg) {
      alert(errMsg);
    }
  });
}

function predictLinesResult(data) {
  for (i = 0; i < data.length; i++) {
    line_text = data[i]
    line = document.createElement("input");
    line.className = 'turing-line';
    line.type = 'text';
    line.value = line_text;
    line.id = 'line-' + ids[i];
    lines.push(line)

    linesBox = document.getElementById("lines-box")
    linesBox.appendChild(line);

    // Highlight box associated to line
    $(line).hover(
      function () {
        line = $(this)
        id = line.attr('id')
        id_num = id.split("-")[1]
        box = $("#box-" + id_num)
        line.addClass('hover')
        box.addClass('hover')
      },
      function () {
        line = $(this)
        id = line.attr('id')
        id_num = id.split("-")[1]
        box = $("#box-" + id_num)
        line.removeClass('hover')
        box.removeClass('hover')
      }
    )
  }
  $("#loader").css("visibility", "hidden");
}

function clearAll() {
  clearBoxes();
  clearLines();
}

function clearBoxes() {
  for (box of boxes) {
    box.remove()
  }
  boxes = []
}

function clearLines() {
  for (line of lines) {
    line.remove()
  }
  lines = []
}

function saveDataset() {

  if (stage < 3) {
    alert("Please get and adjust the text predictions first!");
    return;
  }

  stage = 4;

  data = [];
  for (i = 0; i < lines.length; i++) {
    line = lines[i]
    line_text = line.value;
    id = $(line).attr('id').split('-')[1];
    box = document.getElementById('box-' + id);
    var _top = Math.round(box.style.top.split("px")[0] / total_proportion - topOffset);
    var left = Math.round(box.style.left.split("px")[0] / total_proportion - leftOffset);
    var bottom = _top + (Math.round(box.style.height.split("px")[0] / total_proportion));
    var right = left + (Math.round(box.style.width.split("px")[0] / total_proportion))
    tuple = [left, _top, right, bottom]
    data.push({ 'coords': tuple, 'label': line_text });
  }
  $.ajax({
    type: "POST",
    url: URL + "save_data?file=" + file.name,
    data: JSON.stringify(data),
    contentType: "application/json; charset=utf-8",
    success: function (data) {
      alert("Data has been saved.")
    },
    failure: function (errMsg) {
      alert(errMsg);
    }
  });
}

function appendLines() {

  if (stage < 3) {
    alert("Please get and adjust the text predictions first!");
    return;
  }

  stage = 4;

  var filename = prompt("File to append lines:", "filename.txt");
  var lineTexts = [];
  for (let line of lines) {
    lineTexts.push(line.value)
  }
  $.ajax({
    type: "POST",
    url: URL + "append_turing_lines?file=" + filename,
    data: JSON.stringify(lineTexts),
    contentType: "application/json; charset=utf-8",
    success: function (data) {
      alert("Saved correctly.")
    },
    failure: function (errMsg) {
      alert(errMsg);
    }
  });
}

function clusterSortPoints(points) {
  var threshold = 70;
  var columns = [];

  points = points.sort((point1, point2) => point1[0] - point2[0])

  var current_col = [];
  var col_average = 0;

  // Loop over points
  for (let point of points) {

    // If this is the first point, add it and continue
    if (current_col.length == 0) {
      current_col.push(point);
      continue;
    }

    // Compute current column X coordinate average
    col_average = 0;
    for (let point of current_col) {
      col_average += point[0];
    }
    col_average /= current_col.length;

    if (point[0] - col_average > threshold) {
      // If point is too distant, create new column
      columns.push(current_col);
      current_col = [point];
    } else {
      // Else add to current column
      current_col.push(point);
    }
  }

  // Append the last column as well
  columns.push(current_col)

  // Create final sorted array
  var sortedPoints = [];
  for (let column of columns) {
    let sortedCol = column.sort((point1, point2) => point1[1] - point2[1]);
    sortedPoints.push(...sortedCol);
  }

  return sortedPoints;
}


// Draggable elements
// Source: https://www.w3schools.com/howto/tryit.asp?filename=tryhow_js_draggable
function dragElement(elmnt) {
  var pos1 = 0, pos2 = 0, pos3 = 0, pos4 = 0;
  elmnt.children[0].onmousedown = dragMouseDown;

  function dragMouseDown(e) {
    e = e || window.event;
    e.preventDefault();
    // get the mouse cursor position at startup:
    pos3 = e.clientX;
    pos4 = e.clientY;
    document.onmouseup = closeDragElement;
    // call a function whenever the cursor moves:
    document.onmousemove = elementDrag;
  }

  function elementDrag(e) {
    e = e || window.event;
    e.preventDefault();
    // calculate the new cursor position:
    pos1 = pos3 - e.clientX;
    pos2 = pos4 - e.clientY;
    pos3 = e.clientX;
    pos4 = e.clientY;
    // set the element's new position:
    elmnt.style.top = (elmnt.offsetTop - pos2) + "px";
    elmnt.style.left = (elmnt.offsetLeft - pos1) + "px";
  }

  function closeDragElement() {
    /* stop moving when mouse button is released:*/
    document.onmouseup = null;
    document.onmousemove = null;
  }
}