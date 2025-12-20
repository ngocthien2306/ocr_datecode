import api from './http';
import { authAPI } from './auth';
import { usersAPI } from './users';
import recipesAPI, { receiptsAPI } from './recipes';
import { camerasAPI } from './cameras';

export { authAPI, usersAPI, recipesAPI, receiptsAPI, camerasAPI };
export default api;
