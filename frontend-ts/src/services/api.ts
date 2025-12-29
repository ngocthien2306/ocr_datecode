import api from './http';
import { authAPI } from './auth';
import { usersAPI } from './users';
import recipesAPI, { receiptsAPI } from './recipes';
import { camerasAPI } from './cameras';
import { uploadAPI } from './upload';

export { authAPI, usersAPI, recipesAPI, receiptsAPI, camerasAPI, uploadAPI };
export default api;
