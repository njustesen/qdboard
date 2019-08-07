
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

appControllers.controller('RunShowCtrl', ['$scope', '$window', 'RunService',
    function RunShowCtrl($scope, $window, RunService) {

        $scope.getRun = function getRun(id){
            RunService.get(id).success(function(data) {
                 return data;
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.getArchive = function getArchive(id){
            RunService.getArchive(id).success(function(data) {
                 return data;
            }).error(function(status, data) {
                console.log(status);
                console.log(data);
            });
        };

        $scope.run_id = $routeParams.id;
        $scope.run = $scope.getRun($scope.run_id);
        $scope.archive = $scope.getArchive($scope.run_id);

    }

]);
