using Microsoft.AspNetCore.Mvc;
using Palmprint.Application.DTOs;
using Palmprint.Application.Interfaces;
using Palmprint.Api.Models;

namespace Palmprint.Api.Controllers;

[ApiController]
[Route("api/enrollment")]
public class EnrollmentController : ControllerBase
{
    private readonly IEnrollmentService _enrollmentService;

    public EnrollmentController(IEnrollmentService enrollmentService)
    {
        _enrollmentService = enrollmentService;
    }

    [HttpPost("enroll")]
    [Consumes("multipart/form-data")]
    public async Task<IActionResult> Enroll([FromForm] EnrollFormRequest form)
    {
        if (form.Image == null || form.Image.Length == 0)
        {
            return BadRequest("Image is required.");
        }

        using var memoryStream = new MemoryStream();

        await form.Image.CopyToAsync(memoryStream);

        var request = new EnrollRequest
        {
            TenantId = form.TenantId,
            FullName = form.FullName,
            ExternalId = form.ExternalId,
            ImageBytes = memoryStream.ToArray()
        };

        var response = await _enrollmentService.EnrollAsync(request);

        return Ok(response);
    }
}