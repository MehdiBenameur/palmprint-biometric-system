using Palmprint.Application.DTOs;

namespace Palmprint.Application.Interfaces;

public interface IEnrollmentService
{
    Task<EnrollResponse> EnrollAsync(EnrollRequest request);
}