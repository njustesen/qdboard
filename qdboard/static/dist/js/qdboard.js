'use strict';

var app = angular.module('app', ['ngRoute', 'ngSanitize', 'appControllers', 'appServices', 'appDirectives', 'appFilters']);

var appServices = angular.module('appServices', []);
var appControllers = angular.module('appControllers', []);
var appDirectives = angular.module('appDirectives', []);
var appFilters = angular.module('appFilters', []);

var options = {};
options.api = {};
options.api.base_url = "http://127.0.0.1:5000";


app.config(['$locationProvider', '$routeProvider', 
  function($location, $routeProvider) {
    $routeProvider.
        when('/', {
            templateUrl: 'static/partials/run.list.html',
            controller: 'RunListCtrl'
        }).
        when('/run/create', {
            templateUrl: 'static/partials/run.create.html',
            controller: 'RunListCtrl',
            access: { requiredAuthentication: true }
        }).
        when('/run/:id', {
            templateUrl: 'static/partials/run.show.html',
            controller: 'RunShowCtrl',
            access: { requiredAuthentication: true }
        }).
        otherwise({
            redirectTo: '/'
        });
}]);



appControllers.controller('RunListCtrl', ['$scope', '$window', 'RunService',
    function RunListCtrl($scope, $window, RunService) {
        $scope.runs = [];

        RunService.findAll().success(function(data) {
            $scope.runs = data;
        });

        $scope.loadRun = function loadRun(id){
            RunService.get(id).success(function(data) {
                 $window.location.href = '/#/run/show/' + data.run_id
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.deleteRun = function deleteRun(id) {
            if (id !== undefined) {
                RunService.delete(id).success(function(data) {
                    return data;
                }).error(function(status, data) {
                    console.log(status);
                    console.log(data);
                });
            }
        };

    }

]);


appControllers.controller('RunCreateCtrl', ['$scope', '$window', 'RunService',
    function RunCreateCtrl($scope, $window, RunService) {

        $scope.loadRun = function loadRun(id){
            RunService.get(id).success(function(data) {
                 $window.location.href = '/#/run/show/' + data.run_id
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };
        $scope.create = function save(run) {
            //var content = $('#textareaContent').val();
            RunService.create(run).success(function(data) {
                $window.location.href = '/#/run/show/' + data.run_id
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

    }

]);

appControllers.controller('RunShowCtrl', ['$scope', '$routeParams', '$window', 'RunService',
    function RunShowCtrl($scope, $routeParams, $window, RunService) {

        $scope.getRun = function getRun(id){
            RunService.get(id).success(function(data) {
                $scope.run = data;
                $scope.viridis = d3.scaleSequential().domain([$scope.run.problem.min_fit, $scope.run.problem.max_fit]).interpolator(d3.interpolateViridis);
                $scope.loadArchive($scope.run_id);
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.checkForReload = function checkForReload(time){
            setTimeout(function(){
                if (!$scope.reloading){
                    $scope.loadArchive($scope.run_id);
                    $scope.checkForReload(time);
                }
            }, time);
        };

        $scope.loadArchive = function loadArchive(id){
            $scope.reloading = true;
            RunService.getArchive(id).success(function(data) {
                $scope.archive = data;
                $scope.drawMap($scope.run, $scope.archive);
                $scope.reloading = false;
                $scope.checkForReload(0);
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.solutionToShow = function(){
            if ($scope.solutionInFocus !== null){
                return $scope.solutionInFocus;
            } else if ($dcope.solutionClicked !== null){
                return $scope.solutionClicked;
            }
            return null;
        };

        // Color scale
        $scope.radius = 1;

        $scope.drawMap = function drawMap(run, archive) {

            $('#map').empty();

            // set the dimensions and margins of the graph
            var margin = {top: 10, right: 30, bottom: 30, left: 60},
                width = 560 - margin.left - margin.right,
                height = 500 - margin.top - margin.bottom;

            // append the svg object to the body of the page
            var svg = d3.select("#map")
                .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            //Read the data
            var solutions = [];
            for (var c = 0; c < archive.cells.length; c++) {
                let cell = archive.cells[c];
                for (var s = 0; s < cell.solutions.length; s++) {
                    let solution = cell.solutions[s];
                    solutions.push(solution);
                }
            }

            // Add X axis
            var x = d3.scaleLinear()
                .domain([archive.dimensions[0].min_value, archive.dimensions[0].max_value])
                .range([ 0, width ]);
            svg.append("g")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.axisBottom(x));

            // text label for the x axis
            svg.append("text")
                .attr("transform",
                    "translate(" + (width/2) + " ," + (height + margin.top + 20) + ")")
                .style("text-anchor", "middle")
                .text(archive.dimensions[0].name);

            // Add Y axis
            var y = d3.scaleLinear()
                .domain([archive.dimensions[1].min_value, archive.dimensions[1].max_value])
                .range([ height, 0]);
            svg.append("g")
                .call(d3.axisLeft(y));

            // text label for the y axis
            svg.append("text")
                .attr("transform", "rotate(-90)")
                .attr("y", 0 - margin.left)
                .attr("x", 0 - (height / 2))
                .attr("dy", "1em")
                .style("text-anchor", "middle")
                .text(archive.dimensions[1].name);

            // gridlines in x axis function
            function make_x_gridlines() {
                return d3.axisBottom(x)
                    .ticks(5)
            }

            // gridlines in y axis function
            function make_y_gridlines() {
                return d3.axisLeft(y)
                    .ticks(5)
            }

            // add the X gridlines
            svg.append("g")
                .attr("class", "grid")
                .attr("transform", "translate(0," + height + ")")
                .call(make_x_gridlines()
                    .tickSize(-height)
                    .tickFormat("")
                );

            // add the Y gridlines
            svg.append("g")
                .attr("class", "grid")
                .call(make_y_gridlines()
                    .tickSize(-width)
                    .tickFormat("")
                );

            // Draw cells
            var scaleX = function(x){
                let r = (x - archive.dimensions[0].min_value) / (archive.dimensions[0].max_value - archive.dimensions[0].min_value);
                return Math.max(Math.min(r * width, width), 0);
            };

            var scaleY = function(x){
                let r = (x - archive.dimensions[1].min_value) / (archive.dimensions[1].max_value - archive.dimensions[1].min_value);
                return Math.max(Math.min(height - (r * height), height), 0);
            };

            svg.selectAll("polygon")
                .data(archive.cells)
                .enter().append("polygon")
                .attr("points",function(d) {
                        return d.points.map(function(d) { return [scaleX(d[0]), scaleY(d[1])].join(","); }).join(" ");})
                    .attr("fill", function(d){
                        if (d.fitness_max === null){
                            return 'white';
                        }
                        return $scope.viridis(d.fitness_max)
                    })
                    .attr("stroke", 'black' )
                    .attr("stroke-width", 0.5)
                    .on("mouseover", handleMouseOver)
                    .on("mouseout", handleMouseOut)
                    .on("click", handleMouseClick);

            // Add dots
            svg.append('g')
                .selectAll("dot")
                .data(solutions)
                .enter()
                .append("circle")
                    .attr("cx", function (d) { return x(d.behavior[0]); } )
                    .attr("cy", function (d) { return y(d.behavior[1]); } )
                    .attr("r", $scope.radius)
                    .attr("fill", 'black' )

        };

        // Create Event Handlers for mouse
        function handleMouseClick(d, i) {  // Add interactivity

            $scope.$apply(function () {
                if (d.solutions[0] === $scope.solutionClicked){
                    $scope.solutionClicked = null;
                } else {
                    $scope.solutionClicked = d.solutions[0];
                    // Use D3 to select element, change color and size
                }
            });
        }

        // Create Event Handlers for mouse
        function handleMouseOver(d, i) {  // Add interactivity
            $scope.$apply(function () {
                $scope.solutionInFocus = d.solutions[0];
            });
        }

        function handleMouseOut(d, i) {
            $scope.$apply(function () {
                $scope.solutionInFocus = null;
            });
            //console.log($scope.solutionInFocus);
        }

        $scope.reloading = false;
        $scope.solutionInFocus = null;
        $scope.solutionClicked = null;
        $scope.run_id = $routeParams.id;
        $scope.getRun($scope.run_id);

    }

]);

appDirectives.directive('displayMessage', function() {
	return {
		restrict: 'E',
		scope: {
        	messageType: '=type',
        	message: '=data'
      	},
		template: '<div class="alert {{messageType}}">{{message}}</div>',
		link: function (scope, element, attributes) {
            scope.$watch(attributes, function (value) {
            	console.log(attributes);
            	console.log(value);
            	console.log(element[0]);
                element[0].children.hide(); 
            });
        }
	}
});
appFilters.filter('range', function() {
  return function(input, total) {
    total = parseInt(total);
    for (var i=0; i<total; i++)
      input.push(i);
    return input;
  };
});

app.filter('reverse', function() {
  return function(items) {
    return items.slice().reverse();
  };
});

appServices.factory('RunService', function($http) {
    return {
        get: function(id) {
            return $http.get(options.api.base_url + '/runs/' + id);
        },

        getArchive: function(id) {
            return $http.get(options.api.base_url + '/runs/' + id + '/archive');
        },
        
        findAll: function() {
            return $http.get(options.api.base_url + '/runs/');
        },

        delete: function(id) {
            return $http.delete(options.api.base_url + '/runs/' + id + "/delete");
        },

        create: function(run) {
            return $http.put(options.api.base_url + '/runs/create', {'run': run});
        }

    };
});
