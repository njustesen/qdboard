
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
                $scope.loadArchive($scope.run_id);
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.loadArchive = function loadArchive(id){
            RunService.getArchive(id).success(function(data) {
                $scope.archive = data;
                $scope.drawMap($scope.run, $scope.archive);
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.drawMap = function drawMap(run, archive) {

            // set the dimensions and margins of the graph
            var margin = {top: 10, right: 30, bottom: 30, left: 60},
                width = 460 - margin.left - margin.right,
                height = 400 - margin.top - margin.bottom;

            // append the svg object to the body of the page
            var svg = d3.select("#map")
                .append("svg")
                    .attr("width", width + margin.left + margin.right)
                    .attr("height", height + margin.top + margin.bottom)
                .append("g")
                    .attr("transform", "translate(" + margin.left + "," + margin.top + ")");

            //Read the data
            var solutions = [];
            var regions = [];
            for (var c = 0; c < archive.cells.length; c++) {
                let cell = archive.cells[c];
                for (var s = 0; s < cell.solutions.length; s++) {
                    let solution = cell.solutions[s];
                    solutions.push(solution);
                }
                regions.push(cell.points);
            }
            for (let cell in archive.cells){
                regions.push(cell.points);
                for (let s in cell.solutions) {
                    solutions.push(s.behavior);
                }
            }

            // Add X axis
            var x = d3.scaleLinear()
                .domain([archive.dimensions[0].min_value, archive.dimensions[0].max_value])
                .range([ 0, width ]);
            svg.append("g")
                .attr("transform", "translate(0," + height + ")")
                .call(d3.axisBottom(x));

            // Add Y axis
            var y = d3.scaleLinear()
                .domain([archive.dimensions[1].min_value, archive.dimensions[1].max_value])
                .range([ height, 0]);
            svg.append("g")
                .call(d3.axisLeft(y));

            // Color scale
            var viridis = d3.scaleSequential().domain([run.problem.min_fit, run.problem.max_fit]).interpolator(d3.interpolateViridis);

            // Add dots
            svg.append('g')
                .selectAll("dot")
                .data(solutions)
                .enter()
                .append("circle")
                    .attr("cx", function (d) { return x(d.behavior[0]); } )
                    .attr("cy", function (d) { return y(d.behavior[1]); } )
                    .attr("r", 1.5)
                    .attr("fill", function(d){return viridis(d.fitness)} )

        };

        $scope.run_id = $routeParams.id;
        $scope.getRun($scope.run_id);

    }

]);
