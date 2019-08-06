
appControllers.controller('RunCtrl', ['$scope', '$window', 'RunService',
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

        $scope.getRun = function getRun(id){
            RunService.get(id).success(function(data) {
                 return data;
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
