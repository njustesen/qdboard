
appServices.factory('RunService', function($http) {
    return {
        get: function(id) {
            return $http.get(options.api.base_url + '/runs/' + id);
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
